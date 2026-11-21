"""Что есть в сцене:
  * освещение: 6 тёплых подвесных ламп (модель Блинна-Фонга, затухание) + слабый
    холодный свет из окон + полусферическое окружающее освещение;
  * тени: мягкие (PCF) тени от каждой лампы через кубические карты теней;
  * текстуры: процедурные (плитка, кирпич, штукатурка, дерево, скатерть, картины,
    меню, вывеска, ночной вид за окном) + mip-mapping и анизотропная фильтрация;
  * материалы: цвет, зеркальная составляющая, блеск, свечение (emissive);
  * динамическая камера: свободный полёт (W/A/S/D, Q/E, мышь, Shift) и
    автоматический облёт зала (клавиша F).

Управление: W/A/S/D - ходьба, Q/E - вниз/вверх, мышь - поворот, Shift - быстрее,
F - автооблёт вкл/выкл, R - вернуть камеру, Esc - выход.
"""
import ctypes
import math
import sys

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH, HEIGHT = 1280, 720
NUM_LIGHTS = 6
SHADOW_RES = 768       # разрешение грани кубической карты теней
SHADOW_FAR = 25.0      # дальность света лампы

ROOM_W, ROOM_D, ROOM_H = 16.0, 12.0, 3.6   # размеры зала
HW, HD = ROOM_W / 2, ROOM_D / 2

# ----------------------------------------------------------------------------
# Матрицы (математическая запись: M @ v, столбцы-векторы; в GL грузим с transpose)
# ----------------------------------------------------------------------------


def translate(x, y, z):         # сдвиг
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):             # масштабирование
    return np.diag([x, y, z, 1.0])


def rot_y(a):                   # поворот вокруг вертикальной оси Y
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(a):                   # поворот вокруг оси Z 
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def normalize(v):                           # нормировка
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def look_at(eye, target, up=(0, 1, 0)):     # матрица вида (камеры)
    eye = np.asarray(eye, float)
    f = normalize(np.asarray(target, float) - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


# ----------------------------------------------------------------------------
# Шейдеры
# ----------------------------------------------------------------------------

SHADOW_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 lightVP;
uniform mat4 model;
out vec3 WorldPos;
void main() {
    vec4 w = model * vec4(aPos, 1.0);
    WorldPos = w.xyz;
    gl_Position = lightVP * w;
}
"""

SHADOW_FS = """
#version 330 core
in vec3 WorldPos;
uniform vec3 lightPos;
uniform float farPlane;
layout(location = 0) out float dist;
void main() { dist = length(WorldPos - lightPos) / farPlane; }
"""

MAIN_VS = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform vec2 uvScale;
out vec3 FragPos;
out vec3 Normal;
out vec2 UV;
void main() {
    vec4 world = model * vec4(aPos, 1.0);
    FragPos = world.xyz;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    UV = aUV * uvScale;
    gl_Position = proj * view * world;
}
"""

MAIN_FS = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec2 UV;
out vec4 FragColor;

uniform sampler2D tex;
uniform vec3 viewPos;
uniform vec3 sunDir;        // направление на "окна" (слабый холодный свет, без теней)
uniform vec3 sunColor;
uniform vec3 pointPos[6];
uniform vec3 pointColor;
uniform vec3 skyAmbient;
uniform vec3 groundAmbient;
uniform vec3 fogColor;
uniform vec3 matColor;
uniform float matSpec;
uniform float matShin;
uniform vec3 emissive;
uniform float glow;
uniform float farPlane;
uniform samplerCube shadow0;
uniform samplerCube shadow1;
uniform samplerCube shadow2;
uniform samplerCube shadow3;
uniform samplerCube shadow4;
uniform samplerCube shadow5;

// 9 смещений — центр + 8 углов куба (±1, ±1, ±1). Нужны для мягких теней
const vec3 OFFS[9] = vec3[](
    vec3(0, 0, 0),
    vec3(1, 1, 1), vec3(1, 1, -1), vec3(1, -1, 1), vec3(1, -1, -1),
    vec3(-1, 1, 1), vec3(-1, 1, -1), vec3(-1, -1, 1), vec3(-1, -1, -1));

// доля затенённости точки от лампы lp (0 - освещена, 1 - в тени)
float shadowCube(samplerCube sm, vec3 lp, vec3 N) {
    vec3 fl = FragPos - lp;     // вектор от лампы к точке
    float d = length(fl);       // реальное расстояние от лампы до точки
    if (d >= farPlane) return 0.0;              // дальше света нет, тени не считаем
    float cosT = max(dot(N, -fl / d), 0.0);     // косинус угла между нормалью и направлением на лампу
    float radius = 0.03 + d * 0.01;             // радиус размытия, растёт с расстоянием
    float bias = max(0.12 * (1.0 - cosT), 0.04) + radius * 0.5;         // запас, чтобы поверхность не затеняла саму себя
    float s = 0.0;
    //
    for (int i = 0; i < 9; ++i) {
        // прочитать из карты расстояние до ближайшего препятствия в чуть сдвинутом направлении 
        // и вернуть из доли обратно в метры
        float closest = texture(sm, fl + OFFS[i] * radius).r * farPlane;
        // если наша точка дальше ближайшего препятствия, она в тени
        s += (d - bias > closest) ? 1.0 : 0.0;
    }
    return s / 9.0;     // среднее по 9 пробам
}

// вклад одной лампы (модель Блинна-Фонга)
vec3 pointLight(vec3 lp, float sh, vec3 N, vec3 V, vec3 albedo) {
    vec3 Lp = lp - FragPos;
    float d = length(Lp);       
    Lp /= d;                // единичный вектор на лампу
    float att = 1.0 / (1.0 + 0.09 * d + 0.025 * d * d);     // квадратичное затухание: свет слабеет с расстоянием 
    float dp = max(dot(N, Lp), 0.0);                        // диффузная составляющая
    vec3 Hp = normalize(Lp + V);                            // «серединный вектор» между направлением на лампу и на камеру
    // блик. Чем больше matShin, тем меньше и резче пятно
    float sp = (dp > 0.0) ? pow(max(dot(N, Hp), 0.0), matShin) * matSpec : 0.0;
    // Умножаем свет лампы на видимость (1.0 - sh), затухание att и складывает диффузный цвет поверхности с бликом
    return (1.0 - sh) * att * pointColor * (dp * albedo + sp);
}

void main() {
    // Цвет поверхности. 
    // Текстуры хранятся в sRGB, а свет считаем в линейном пространстве, поэтому переводим степенью 2.2
    vec3 albedo = pow(texture(tex, UV).rgb, vec3(2.2)) * pow(matColor, vec3(2.2));
    vec3 N = normalize(Normal);
    vec3 V = normalize(viewPos - FragPos);      // Направление на камеру
    vec3 L = normalize(sunDir);

    // окружающее (полусферическое) освещение
    vec3 col = mix(groundAmbient, skyAmbient, N.y * 0.5 + 0.5) * albedo;
    // для нормалей вниз берётся groundAmbient, для вверх — skyAmbient

    // слабый холодный свет из окон (без теней)
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = (diff > 0.0) ? pow(max(dot(N, H), 0.0), matShin) * matSpec : 0.0;
    col += sunColor * (diff * albedo + spec);

    // лампы (каждая со своей картой теней)
//LAMPS//
    col += emissive + glow * albedo;   // светящиеся объекты

    // лёгкая дымка
    float dist = length(viewPos - FragPos);
    float f = 1.0 - exp(-pow(dist * 0.01, 2.0));
    col = mix(col, fogColor, f);

    // тональная коррекция + гамма
    col = vec3(1.0) - exp(-col * 1.1);      // сжимает яркие значения в диапазон 0..1 без резкого пересвета
    col = pow(col, vec3(1.0 / 2.2));        // Обратный переход в sRGB для монитора
    FragColor = vec4(col, 1.0);
}
"""

MAIN_FS = MAIN_FS.replace("//LAMPS//", "\n".join(
    f"    col += pointLight(pointPos[{i}], shadowCube(shadow{i}, pointPos[{i}], N), N, V, albedo);"
    for i in range(6)))

"""f"    col += pointLight(pointPos[{i}], 0.0, N, V, albedo) // без отрисовка теней (и света ваще), для отладки"""

# Превращаем две строки с исходником в готовую программу видеокарты
def build_program(vs, fs):
    # Сливаем два шейдера в одну программу (линковка). Вовзращая ID
    return shaders.compileProgram(
        shaders.compileShader(vs, GL_VERTEX_SHADER),        # Компилируем вершинный шейдер в объект шейдера видеокарты
        shaders.compileShader(fs, GL_FRAGMENT_SHADER),      # Компилируем фрагментный шейдер
        validate=False,                                     # Отключаем проверку в момент создания
    )


class Uniforms:
    def __init__(self, prog):
        self.prog = prog
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            self.cache[name] = glGetUniformLocation(self.prog, name)    # Запрашиваем ID переменной у видеокарты. Заносим в кэш
        return self.cache[name]

# Передаём одну матрицу 4×4 (1 — количество матриц)
def set_mat4(loc, m):
    glUniformMatrix4fv(loc, 1, GL_TRUE, np.asarray(m, dtype=np.float32))
    # GL_TRUE просит драйвер транспонировать матрицу при загрузке


# ----------------------------------------------------------------------------
# Геометрия: формат вершины = pos(3) normal(3) uv(2)
# ----------------------------------------------------------------------------


class Mesh:
    def __init__(self, verts, idx):
        # Приводим массивы вершин и индексов к строгим типам данных, которые требует видеокарта
        verts = np.asarray(verts, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.uint32)
        # Сохраняем общее количество индексов, чтобы знать, сколько точек нужно будет нарисовать при вызове draw()
        self.count = len(idx)  
        # Создаёт и активирует VAO (Vertex Array Object) — объект-контейнер, который запоминает всю дальнейшую настройку буферов и атрибутов вершин
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # Создаёт VBO (Vertex Buffer Object) и копирует туда массив с данными всех вершин (verts) в видеопамять
        # Флаг GL_STATIC_DRAW подсказывает видеокарте, что геометрия не будет меняться каждый кадр
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        # Создаёт EBO (Element Buffer Object) и загружает туда массив индексов (idx)
        # Это позволяет не дублировать общие вершины соседних треугольников, а ссылаться на них по номерам
        # Привязка кэшируется внутри текущего VAO
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        # Шаг между вершинами в байтах
        stride = 8 * 4
        # Включаем атрибут с номером 0 (позиция вершины X, Y, Z в шейдере):
        # берёт 3 числа типа GL_FLOAT с шагом 32 байта, начиная с самого начала вершины (смещение 0 байт)
        # (совпадает с layout(location = 0) в шейдере)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # Включаем атрибут с номером 1 (вектор нормали NX, NY, NZ для расчёта освещения): 
        # берёт 3 числа типа GL_FLOAT со смещением 12 байт
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        # Включаем атрибут с номером 2 (текстурные координаты U, V): 
        # берёт 2 числа типа GL_FLOAT со смещением 24 байта
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        # Отвязываем VAO после завершения настройки, чтобы не испортить в будущем
        glBindVertexArray(0)

    def draw(self):
        # Включает сохранённый ранее VAO
        glBindVertexArray(self.vao)
        # Даёт видеокарте команду нарисовать self.count вершин, собирая из них треугольники (GL_TRIANGLES) по индексам типа GL_UNSIGNED_INT из привязанного EBO 
        # (None означает начинать с нулевого элемента буфера)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, None)


_box_cache = {}


def box(w, h, d):
    """Параллелепипед с центром в начале координат.
    UV заданы в метрах -> текстура тайлится независимо от размеров."""
    key = (round(w, 4), round(h, 4), round(d, 4))
    if key in _box_cache:
        return _box_cache[key]
    faces = [
        # Грань куба в формате (нормаль, 4 угла (front-facing), масштаб текстуры)
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], (w, h)),
        ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)], (w, h)),
        ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)], (d, h)),
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)], (d, h)),
        ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)], (w, d)),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], (w, d)),
    ]
    # Стандартные текстурные координаты для 4 углов прямоугольника (левый-низ, правый-низ, правый-верх, левый-верх)
    uvc = [(0, 0), (1, 0), (1, 1), (0, 1)]
    verts, idx = [], []
    for n, corners, (su, sv) in faces:
        # Запоминаем в base номер первой вершины текущей грани (0, 4, 8, 12, 16, 20)
        base = len(verts)
        for c, uv in zip(corners, uvc):
            # Формируем для каждого массив из 8 чисел (позиция X, Y, Z, нормаль NX, NY, NZ, текстурные координаты U, V)
            verts.append([c[0] * w / 2, c[1] * h / 2, c[2] * d / 2, *n, uv[0] * su, uv[1] * sv])
        # Разбиваем четырехугольную грань на два треугольника: первый (0 -> 1 -> 2) и второй (0 -> 2 -> 3)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    # Создает из собранных 24 вершин и 36 индексов объект Mesh
    m = Mesh(verts, idx)
    _box_cache[key] = m
    return m


def make_panel():
    """Вертикальный квадрат 1x1 в плоскости XY, нормаль +Z, UV 0..1."""
    verts = [
        [-0.5, -0.5, 0, 0, 0, 1, 0, 0],
        [0.5, -0.5, 0, 0, 0, 1, 1, 0],
        [0.5, 0.5, 0, 0, 0, 1, 1, 1],
        [-0.5, 0.5, 0, 0, 0, 1, 0, 1],
    ]
    return Mesh(verts, [0, 1, 2, 0, 2, 3])


def make_sphere(r=1.0, stacks=18, slices=28):
# stacks — число «широтных колец», slices — «долготных долей»
    verts, idx = [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks              # Вычисляем полярный угол фи от 0 до π (от северного полюса к южному)
        for j in range(slices + 1):
            th = 2 * math.pi * j / slices       # Вычисляет азимутальный угол тета от 0 до 2π (полный оборот 360 градусов)
            # Переводит сферические координаты на единичной сфере в декартовы. 
            # Так как радиус этой сферы равен 1, точка на поверхности одновременно является 
            # единичным вектором нормали n (направленным строго от центра сферы наружу)
            n = (math.sin(phi) * math.cos(th), math.cos(phi), math.sin(phi) * math.sin(th))
            # Записываем 8 параметров вершины: 
            # масштабируем единичный вектор на нужный радиус r, получая реальные координаты 
            # распаковываем 3 компонента нормали 
            # задаём UV-координаты
            verts.append([n[0] * r, n[1] * r, n[2] * r, *n, j / slices * 2, i / stacks])
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j                    # Вычисляем индекс верхней левой вершины текущей ячейки
            b = a + slices + 1                          # Вычисляем индекс нижней левой вершины текущей ячейки
            idx += [a, b, a + 1, a + 1, b, b + 1]       # Разбиваем ячейку на два треугольника: верхний (a -> b -> a+1) и нижний (a+1 -> b -> b+1)
    return Mesh(verts, idx)


def make_cylinder(r=1.0, h=1.0, seg=28):
    """Цилиндр вдоль оси Y, центр в начале координат."""
    verts, idx = [], []
    for j in range(seg + 1):
        th = 2 * math.pi * j / seg
        c, s = math.cos(th), math.sin(th)           # Они задают направление от центра наружу в горизонтальной плоскости XZ
        # Добавляет нижнюю точку на окружности 
        verts.append([r * c, -h / 2, r * s, c, 0, s, j / seg * 2, 0])
        # Добавляет парную ей верхнюю точку прямо над ней
        verts.append([r * c, h / 2, r * s, c, 0, s, j / seg * 2, 1])
    for j in range(seg):
        a = 2 * j               # Берём индекс нижней вершины текущего вертикального ребра
        idx += [a, a + 2, a + 1, a + 1, a + 2, a + 3]      
    # Запускаем цикл дважды: сначала для верхней крышки, потом для нижней
    for sign in (1, -1):
        base = len(verts)       # Запоминает индекс центральной вершины текущей крышки
        # Создаёт точку ровно в центре крышки с вертикальной нормалью
        verts.append([0, sign * h / 2, 0, 0, sign, 0, 0.5, 0.5])
        # Снова проходим по окружности и создаёт точки на ободе крышки
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            c, s = math.cos(th), math.sin(th)
            verts.append([r * c, sign * h / 2, r * s, 0, sign, 0, 0.5 + 0.5 * c, 0.5 + 0.5 * s])        # UV крышки: 0.5 + 0.5*cos, 0.5 + 0.5*sin — круг внутри квадрата текстуры
        # Соединяет центральную точку (base) с каждыми двумя соседними точками на ободе 
        # формируя веер треугольников (Triangle Fan), закрывающий торец цилиндра
        for j in range(seg):
            idx += [base, base + 1 + j, base + 2 + j]
    return Mesh(verts, idx)


# ----------------------------------------------------------------------------
# Процедурные текстуры (Pillow + numpy)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(7)


def get_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def noise_img(size, base, var, blur=0):
    arr = np.array(base, float)[None, None, :] + rng.normal(0, var, (size, size, 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img

# Рисует надпись на изображении так, чтобы геометрический центр текста точно совпал с точкой (cx, cy)
def draw_centered(d, text, font, cx, cy, fill):
    # Вычисляет ограничивающий прямоугольник (bounding box) текста при рисовании в точке (0, 0)
    bb = d.textbbox((0, 0), text, font=font)        # Возвращает кортеж (left, top, right, bottom) в пикселях
    # Считает точную ширину (tw) и высоту (th) надписи
    tw, th = bb[2] - bb[0], bb[3] - bb[1]       
    # Рисуем текст цветом fill, сдвигая левый верхний угол на половину ширины и половину высоты
    # Вычитание bb и bb компенсирует внутренние отступы шрифта сверху и слева
    d.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), text, font=font, fill=fill)

# Создаёт крошечную чисто-белую RGB-картинку размером 4×4 пикселя
def tex_white():
    return Image.new("RGB", (4, 4), (255, 255, 255))
# Умножение на белое не меняет цвет, поэтому объект окрашивается только matColor

# 256×256 = 2×2 плитки по 128 px, светлая и тёмная в шахматном порядке ((i+j) % 2) 
# пол
def tex_floor():
    """Шахматная плитка: 2x2 плитки на текстуру (плитка 0.5 м при 1 текстура/м)."""
    img = Image.new("RGB", (256, 256))
    d = ImageDraw.Draw(img)
    for i in range(2):
        for j in range(2):
            col = (222, 212, 195) if (i + j) % 2 == 0 else (52, 46, 44)
            d.rectangle([i * 128, j * 128, i * 128 + 127, j * 128 + 127], fill=col)
    arr = np.array(img, float) + rng.normal(0, 6, (256, 256, 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    d = ImageDraw.Draw(img)
    for p in (0, 128):
        d.line([(p, 0), (p, 255)], fill=(88, 84, 78), width=2)
        d.line([(0, p), (255, p)], fill=(88, 84, 78), width=2)
    return img

# стены, потолок
def tex_plaster():
    return noise_img(256, (228, 208, 178), 4, 1.0)

# 32 ряда по 8 px. В нечётных рядах смещение на полкирпича (off = -16) — перевязка. 
# Каждый кирпич — прямоугольник, яркость случайно смещается на t
# кирпичная стена
def tex_brick():
    img = Image.new("RGB", (256, 256), (150, 140, 130))
    d = ImageDraw.Draw(img)
    for r in range(32):
        off = 0 if r % 2 == 0 else -16
        for c in range(-1, 9):
            x0 = c * 32 + off
            t = int(rng.integers(-18, 19))
            d.rectangle([x0 + 1, r * 8 + 1, x0 + 30, r * 8 + 6],
                        fill=(max(0, 150 + t), max(0, 72 + t // 2), max(0, 55 + t // 2)))
    return img.filter(ImageFilter.GaussianBlur(0.4))

# Коричневый цвет + сильный шум «по строкам» ((size, 1, 1): одно число на всю строку) + слабый попиксельный
# мебель, панели, балки, стойка
def tex_wood():
    size = 256
    arr = np.array((135, 88, 48), float)[None, None, :] + rng.normal(0, 14, (size, 1, 1)) \
        + rng.normal(0, 4, (size, size, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

# листья растений
def tex_grass():
    return noise_img(256, (50, 100, 42), 22, 0.7)

# 128×128, шахматная раскраска 4×4 клетки по 32 px, красные клетки на белом фоне + шум
# скатерти
def tex_cloth():
    """Скатерть в красно-белую клетку (текстура = 0.5 м, клетка 4x4)."""
    img = Image.new("RGB", (128, 128), (245, 242, 235))
    d = ImageDraw.Draw(img)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                d.rectangle([i * 32, j * 32, i * 32 + 31, j * 32 + 31], fill=(190, 45, 45))
    arr = np.array(img, float) + rng.normal(0, 4, (128, 128, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def tex_night(seed):
    """Ночной город за окном."""
    r = np.random.default_rng(seed)
    W, H = 512, 360
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(10 + 30 * t), int(14 + 28 * t), int(40 + 40 * t)))
    for _ in range(70):
        x, y = int(r.integers(0, W)), int(r.integers(0, H // 2))
        v = int(r.integers(150, 255))
        d.point((x, y), fill=(v, v, v))
    d.ellipse([380 if seed % 2 else 80, 30, 420 if seed % 2 else 120, 70], fill=(235, 235, 210))
    x = 0
    while x < W:
        w = int(r.integers(40, 90))
        h = int(r.integers(90, 230))
        g = int(r.integers(12, 28))
        d.rectangle([x, H - h, x + w, H], fill=(g, g, g + 12))
        for wy in range(H - h + 10, H - 10, 18):
            for wx in range(x + 6, x + w - 10, 14):
                if r.random() < 0.45:
                    d.rectangle([wx, wy, wx + 7, wy + 9], fill=(240, 200, 100))
        x += w + int(r.integers(2, 12))
    return img


def tex_painting(seed):
    r = np.random.default_rng(seed)
    bg = tuple(int(v) for v in r.integers(170, 235, 3))
    img = Image.new("RGB", (256, 256), bg)
    d = ImageDraw.Draw(img)
    for _ in range(9):
        col = tuple(int(v) for v in r.integers(20, 240, 3))
        x0, y0 = int(r.integers(0, 200)), int(r.integers(0, 200))
        x1, y1 = x0 + int(r.integers(30, 120)), y0 + int(r.integers(30, 120))
        if r.random() < 0.5:
            d.rectangle([x0, y0, x1, y1], fill=col)
        else:
            d.ellipse([x0, y0, x1, y1], fill=col)
    return img.filter(ImageFilter.GaussianBlur(0.8))


def tex_menu():
    W, H = 512, 384
    img = Image.new("RGB", (W, H), (30, 48, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, W - 9, H - 9], outline=(220, 220, 210), width=3)
    draw_centered(d, "МЕНЮ", get_font(54), W / 2, 55, (250, 240, 200))
    items = [("Борщ", "250"), ("Паста карбонара", "420"), ("Стейк рибай", "980"),
             ("Салат цезарь", "380"), ("Тирамису", "290")]
    f = get_font(28)
    for i, (n, p) in enumerate(items):
        y = 120 + i * 50
        d.text((40, y), n, font=f, fill=(235, 235, 225))
        d.text((W - 110, y), p + " р", font=f, fill=(235, 235, 225))
        d.line([(40, y + 38), (W - 40, y + 38)], fill=(110, 130, 120), width=1)
    return img


def tex_sign():
    W, H = 512, 96
    img = Image.new("RGB", (W, H), (25, 15, 12))
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, W - 6, H - 6], outline=(255, 180, 90), width=4)
    draw_centered(d, "RESTAURANT", get_font(58), W / 2, H / 2, (255, 215, 140))
    return img


def tex_door(label, base):
    W, H = 256, 400
    img = noise_img(256, base, 5, 0.5).resize((W, H))
    d = ImageDraw.Draw(img)
    dark = tuple(int(v * 0.7) for v in base)
    d.rectangle([30, 30, W - 30, 170], outline=dark, width=5)
    d.rectangle([30, 200, W - 30, 370], outline=dark, width=5)
    draw_centered(d, label, get_font(34), W / 2, 100, (240, 230, 200))
    d.ellipse([W - 52, 210, W - 36, 226], fill=(220, 200, 80))
    return img


def load_texture(img):
    # Гарантирует 3-канальный формат RGB
    # Переворачивает картинку по вертикали
    # В Pillow ось Y идёт сверху вниз, а в текстурных координатах OpenGL (UV) — снизу вверх
    img = img.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size                 # Получает ширину и высоту изображения в пикселях
    tid = glGenTextures(1)          # Создаёт новый объект текстуры в OpenGL и получаем его числовой идентификатор tid
    glBindTexture(GL_TEXTURE_2D, tid)           # Делаем его активным для дальнейшей настройки
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)       # Задаёт выравнивание строк пикселей в памяти по 1 байту
    # Загружаем пиксели (img.tobytes()) 
	# 0 — базовый уровень детализации (mip-уровень 0)
	# GL_RGB — формат хранения внутри видеокарты
	# w, h, 0 — размеры текстуры и ширина рамки (в современном OpenGL рамка всегда 0)
	# GL_RGB, GL_UNSIGNED_BYTE — формат входных данных (3 канала по 1 беззнаковому байту 0..255 на канал)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    # Автоматически генерирует на видеокарте набор уменьшенных копий текстуры (мипмапы)
    glGenerateMipmap(GL_TEXTURE_2D)
    # Задаёт режим повторения текстуры по горизонтали (S) и вертикали (T)
    # если UV-координаты выходят за пределы 0..1
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)        # Благодаря этому работает тайлинг стен и полов в функции box
    # Включает трилинейную фильтрацию при уменьшении текстуры
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    # Включает билинейную фильтрацию при увеличении текстуры
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    try:
        glTexParameterf(GL_TEXTURE_2D, 0x84FE, 8.0)  # анизотропная фильтрация, если есть
    except Exception:
        pass
    return tid


# ----------------------------------------------------------------------------
# Сцена
# ----------------------------------------------------------------------------
objs = []
textures = {}
lamp_positions = []

TABLES = [(x, z) for z in (-0.5, 3.5) for x in (-5.0, 0.0, 5.0)]
LAMP_Y = 2.35


def add(mesh, model, color=(1, 1, 1), tex="white", uvs=(1.0, 1.0), spec=0.08, shin=16.0,
        emis=(0, 0, 0), glow=0.0, cast=True):
    # mesh — готовый объект класса Mesh (коробка, панель, сфера, цилиндр)
    # model — матрица модели 4×4, задающая положение, поворот и масштаб объекта в мире; хранится как float32 
    # color — цвет материала RGB 0..1; умножается на цвет текстуры
    # tex — имя текстуры из словаря textures
    # uvs — множитель масштаба текстурных координат (тайлинг по горизонтали и вертикали)
    # spec — сила зеркального блика (specular) (0 — матовый, 1 — сильный блик)
    # shin — острота блика (shininess); чем выше число, тем меньше и острее блик
    # emis — цвет собственного свечения материала (emissive); (используется для лампочек, экранов или свечей)
    # glow — коэффициент дополнительного яркого свечения/ореола в шейдере
    # cast — отбрасывает ли объект тень

    objs.append((mesh, np.asarray(model, np.float32), color, textures[tex], uvs, spec, shin, emis, glow, cast))


def build_textures():
    textures["white"] = load_texture(tex_white())
    textures["floor"] = load_texture(tex_floor())
    textures["plaster"] = load_texture(tex_plaster())
    textures["brick"] = load_texture(tex_brick())
    textures["wood"] = load_texture(tex_wood())
    textures["grass"] = load_texture(tex_grass())
    textures["cloth"] = load_texture(tex_cloth())
    textures["night0"] = load_texture(tex_night(1))
    textures["night1"] = load_texture(tex_night(2))
    for i in range(3):
        textures[f"paint{i}"] = load_texture(tex_painting(40 + i))
    textures["menu"] = load_texture(tex_menu())
    textures["sign"] = load_texture(tex_sign())
    textures["door_front"] = load_texture(tex_door("ВХОД", (120, 78, 44)))
    textures["door_kitchen"] = load_texture(tex_door("КУХНЯ", (170, 170, 165)))


def build_scene():
    # Один раз создаём единичные меши квадрата, сферы и цилиндра, 
    # чтобы многократно переиспользовать их для разных объектов, меняя только матрицу model
    panel = make_panel()
    sphere = make_sphere()
    cyl = make_cylinder()

    # ---- помощники: base - матрица "на стене", локальный +Z смотрит в зал ----
    def wall_window(base, w=2.0, h=1.4, night=0):
        add(panel, base @ scale(w, h, 1), (1, 1, 1), f"night{night}", (1, 1), 0.0, 8, glow=0.7, cast=False)
        t = 0.1
        for dx, dy, sw, sh in ((0, h / 2 + t / 2, w + 2 * t, t), (0, -h / 2 - t / 2, w + 2 * t, t),
                               (-w / 2 - t / 2, 0, t, h), (w / 2 + t / 2, 0, t, h)):
            add(box(sw, sh, 0.12), base @ translate(dx, dy, 0.05), (0.92, 0.9, 0.85), "white", spec=0.1)
        add(box(0.05, h, 0.06), base @ translate(0, 0, 0.03), (0.92, 0.9, 0.85), "white", spec=0.1)
        add(box(w, 0.05, 0.06), base @ translate(0, 0, 0.03), (0.92, 0.9, 0.85), "white", spec=0.1)
        add(box(w + 0.4, 0.06, 0.25), base @ translate(0, -h / 2 - t - 0.03, 0.1), (0.9, 0.88, 0.82), "white", spec=0.1)

    def wall_art(base, w, h, tex):
        add(box(w + 0.12, h + 0.12, 0.05), base @ translate(0, 0, 0.025), (0.3, 0.2, 0.12), "wood", spec=0.2, shin=24)
        add(panel, base @ translate(0, 0, 0.06) @ scale(w, h, 1), (1, 1, 1), tex, (1, 1), 0.15, 24, cast=False)

    def wall_door(base, w, h, tex):
        add(panel, base @ scale(w, h, 1), (1, 1, 1), tex, (1, 1), 0.3, 32, cast=False)
        t = 0.1
        for sx in (-1, 1):
            add(box(t, h + t, 0.12), base @ translate(sx * (w / 2 + t / 2), t / 2, 0.05), (0.35, 0.22, 0.12), "wood", spec=0.2)
        add(box(w + 2 * t, t, 0.12), base @ translate(0, h / 2 + t / 2, 0.05), (0.35, 0.22, 0.12), "wood", spec=0.2)

    # ---- пол, стены, потолок ----
    add(box(ROOM_W + 0.8, 0.2, ROOM_D + 0.8), translate(0, -0.1, 0), (1, 1, 1), "floor", (1, 1), 0.45, 64)
    add(box(ROOM_W + 0.4, ROOM_H + 0.2, 0.2), translate(0, ROOM_H / 2 + 0.1, -HD - 0.1), (1, 1, 1), "brick", (0.42, 0.42), 0.08, 16)
    add(box(ROOM_W + 0.4, ROOM_H + 0.2, 0.2), translate(0, ROOM_H / 2 + 0.1, HD + 0.1), (0.95, 0.9, 0.82), "plaster", (0.5, 0.5), 0.05, 8)
    for s in (-1, 1):
        add(box(0.2, ROOM_H + 0.2, ROOM_D), translate(s * (HW + 0.1), ROOM_H / 2 + 0.1, 0), (0.95, 0.9, 0.82), "plaster", (0.5, 0.5), 0.05, 8)
    add(box(ROOM_W + 0.4, 0.2, ROOM_D + 0.4), translate(0, ROOM_H + 0.1, 0), (0.92, 0.88, 0.8), "plaster", (0.5, 0.5), 0.03, 8)

    # буазери (левая, правая, передняя)
    for s in (-1, 1):
        add(box(0.06, 1.1, ROOM_D), translate(s * (HW - 0.04), 0.55, 0), (0.8, 0.6, 0.45), "wood", (1, 1), 0.2, 24)
        add(box(0.1, 0.06, ROOM_D), translate(s * (HW - 0.06), 1.13, 0), (0.55, 0.38, 0.25), "wood", (1, 1), 0.3, 32)
    add(box(ROOM_W, 1.1, 0.06), translate(0, 0.55, HD - 0.04), (0.8, 0.6, 0.45), "wood", (1, 1), 0.2, 24)
    add(box(ROOM_W, 0.06, 0.1), translate(0, 1.13, HD - 0.06), (0.55, 0.38, 0.25), "wood", (1, 1), 0.3, 32)

    # потолочные балки (между рядами ламп)
    for bx in (-2.5, 2.5):
        add(box(0.25, 0.2, ROOM_D), translate(bx, ROOM_H - 0.1, 0), (0.5, 0.34, 0.22), "wood", (1, 1), 0.2, 24)

    # ---- окна, двери, картины ----
    
    wall_door(translate(0, 1.1, HD - 0.02) @ rot_y(math.pi), 1.4, 2.2, "door_front")
    wall_art(translate(-2.2, 1.9, HD - 0.02) @ rot_y(math.pi), 0.8, 1.0, "paint0")
    wall_art(translate(2.2, 1.9, HD - 0.02) @ rot_y(math.pi), 0.8, 1.0, "paint1")
    # меню на правой стене
    add(box(1.8, 1.3, 0.05), translate(HW - 0.04, 1.9, -2.5) @ rot_y(-math.pi / 2) @ translate(0, 0, 0.0), (0.35, 0.22, 0.12), "wood", spec=0.2)
    add(panel, translate(HW - 0.07, 1.9, -2.5) @ rot_y(-math.pi / 2) @ scale(1.65, 1.2, 1),
        (1, 1, 1), "menu", (1, 1), 0.05, 8, cast=False)
    # задняя стена: кухня, картина, вывеска
    wall_door(translate(-5.0, 1.1, -HD + 0.02), 1.2, 2.2, "door_kitchen")
    wall_art(translate(-2.6, 1.9, -HD + 0.02), 1.0, 1.0, "paint2")
    add(panel, translate(3.5, 3.1, -HD + 0.02) @ scale(3.2, 0.6, 1), (1, 1, 1), "sign", (1, 1), 0.1, 16,
        emis=(0.25, 0.12, 0.04), glow=1.4, cast=False)

    # ---- барная стойка и полки с бутылками ----
    add(box(8.0, 1.05, 0.8), translate(3.5, 0.525, -3.9), (0.45, 0.28, 0.16), "wood", (1, 1), 0.3, 32)
    add(box(8.3, 0.08, 1.1), translate(3.5, 1.09, -3.85), (0.3, 0.18, 0.1), "wood", (1, 1), 0.7, 96)
    add(cyl, translate(3.5, 0.2, -3.45) @ rot_z(math.pi / 2) @ scale(0.03, 8.0, 0.03),
        (0.8, 0.7, 0.4), "white", spec=0.9, shin=96)
    for sy in (1.35, 1.85, 2.35):
        add(box(8.0, 0.04, 0.35), translate(3.5, sy, -HD + 0.2), (0.35, 0.22, 0.12), "wood", (1, 1), 0.3, 32)
    for sx in (-0.5, 7.5):
        add(box(0.05, 1.2, 0.35), translate(sx, 1.85, -HD + 0.2), (0.35, 0.22, 0.12), "wood", (1, 1), 0.3, 32)
    for sy in (1.85, 2.35):  # подсветка под полками
        add(box(7.8, 0.02, 0.05), translate(3.5, sy - 0.03, -HD + 0.38), (1, 1, 1), "white",
            emis=(2.0, 1.4, 0.6), cast=False)
    br = np.random.default_rng(3)
    for sy in (1.35, 1.85, 2.35):
        for bx in np.arange(-0.8, 7, 0.34):
            col = tuple(float(v) for v in br.uniform(0.1, 0.9, 3))
            hgt = float(br.uniform(0.22, 0.32))
            add(cyl, translate(bx + 0.5, sy + 0.02 + hgt / 2, -HD + 0.2) @ scale(0.055, hgt, 0.055),
                col, "white", spec=1.0, shin=128, cast=False)
            add(cyl, translate(bx + 0.5, sy + 0.02 + hgt + 0.05, -HD + 0.2) @ scale(0.02, 0.1, 0.02),
                col, "white", spec=1.0, shin=128, cast=False)
    # барные стулья
    for sx in (0.5, 2.0, 3.5, 5.0, 6.5):
        add(cyl, translate(sx, 0.72, -2.85) @ scale(0.2, 0.06, 0.2), (0.55, 0.1, 0.1), "white", spec=0.3, shin=32)
        add(cyl, translate(sx, 0.35, -2.85) @ scale(0.03, 0.7, 0.03), (0.15, 0.15, 0.17), "white", spec=0.6, shin=64)
        add(cyl, translate(sx, 0.02, -2.85) @ scale(0.2, 0.04, 0.2), (0.15, 0.15, 0.17), "white", spec=0.6, shin=64)
    # предметы на стойке
    for gx in (1.0, 2.2, 4.8, 6.0):
        add(cyl, translate(gx, 1.13 + 0.07, -3.9) @ scale(0.035, 0.14, 0.035), (0.75, 0.88, 1.0), "white", spec=1.0, shin=128)
    add(box(0.4, 0.25, 0.3), translate(0.2, 1.13 + 0.125, -4.1), (0.2, 0.2, 0.22), "white", spec=0.5, shin=48)  # касса

    # ---- столы, стулья, сервировка, лампы ----
    def chair(x, z, yaw):
        base = translate(x, 0, z) @ rot_y(yaw)          # Задаём локальную систему координат стула на полу
        wood = (0.8, 0.6, 0.45)
        add(box(0.46, 0.05, 0.46), base @ translate(0, 0.45, 0), wood, "wood", (1, 1), 0.2, 24)
        add(box(0.4, 0.04, 0.4), base @ translate(0, 0.495, 0), (0.55, 0.12, 0.1), "white", spec=0.15, shin=16)
        add(box(0.46, 0.4, 0.05), base @ translate(0, 0.78, -0.2), wood, "wood", (1, 1), 0.2, 24)
        for sx in (-0.2, 0.2):
            add(box(0.05, 0.43, 0.05), base @ translate(sx, 0.215, 0.2), wood, "wood", (1, 1), 0.2, 24)
            add(box(0.05, 0.98, 0.05), base @ translate(sx, 0.49, -0.2), wood, "wood", (1, 1), 0.2, 24)

    for tx, tz in TABLES:
        # стол: пьедестал + столешница + скатерть
        add(cyl, translate(tx, 0.02, tz) @ scale(0.28, 0.04, 0.28), (0.15, 0.15, 0.17), "white", spec=0.6, shin=64)
        add(cyl, translate(tx, 0.37, tz) @ scale(0.06, 0.7, 0.06), (0.15, 0.15, 0.17), "white", spec=0.6, shin=64)
        add(box(0.96, 0.04, 0.96), translate(tx, 0.72, tz), (0.8, 0.6, 0.45), "wood", (1, 1), 0.2, 24)
        add(box(1.0, 0.03, 1.0), translate(tx, 0.765, tz), (1, 1, 1), "cloth", (2, 2), 0.05, 8)
        # 4 места: тарелка, бокал, стул
        for dx, dz in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            px, pz = tx + 0.27 * dx, tz + 0.27 * dz
            add(cyl, translate(px, 0.78 + 0.008, pz) @ scale(0.14, 0.016, 0.14), (0.96, 0.96, 0.94), "white", spec=0.8, shin=96)
            gx, gz = px + 0.17 * (-dz), pz + 0.17 * dx
            add(cyl, translate(gx, 0.78 + 0.05, gz) @ scale(0.035, 0.1, 0.035), (0.75, 0.88, 1.0), "white", spec=1.0, shin=128)
            chair(tx + 0.75 * dx, tz + 0.75 * dz, math.atan2(-dx, -dz))
        # ваза с цветком и свеча
        add(cyl, translate(tx, 0.78 + 0.07, tz) @ scale(0.05, 0.14, 0.05), (0.15, 0.5, 0.3), "white", spec=0.9, shin=96)
        add(sphere, translate(tx, 0.78 + 0.21, tz) @ scale(0.07, 0.07, 0.07), (0.85, 0.12, 0.2), "white", spec=0.15, cast=False)
        add(cyl, translate(tx + 0.15, 0.78 + 0.05, tz + 0.15) @ scale(0.03, 0.1, 0.03), (0.95, 0.92, 0.8), "white", spec=0.1)
        add(sphere, translate(tx + 0.15, 0.78 + 0.115, tz + 0.15) @ scale(0.015, 0.025, 0.015),
            (1, 0.9, 0.5), "white", emis=(4.0, 2.5, 0.8), cast=False)
        # подвесная лампа: шнур, плафон, лампочка
        add(cyl, translate(tx, (2.64 + ROOM_H) / 2, tz) @ scale(0.01, ROOM_H - 2.64, 0.01), (0.1, 0.1, 0.1), "white", cast=False)
        add(cyl, translate(tx, 2.55, tz) @ scale(0.3, 0.18, 0.3), (0.12, 0.35, 0.25), "white", spec=0.5, shin=48)
        add(sphere, translate(tx, LAMP_Y, tz) @ scale(0.08, 0.08, 0.08), (1, 0.95, 0.8), "white",
            emis=(2.6, 1.9, 1.0), cast=False)
        lamp_positions.append((tx, LAMP_Y, tz))     # Сам источник света будет в шейдере, а не в сцене, поэтому cast=False

    # ---- растения в кадках ----
    for px, pz in ((-7.2, -5.2), (7.2, 5.2), (-7.2, 5.2), (1.6, 5.3)):
        add(cyl, translate(px, 0.2, pz) @ scale(0.25, 0.4, 0.25), (0.7, 0.35, 0.22), "white", spec=0.1)
        for ox, oy, oz, r in ((0, 0.70, 0, 0.32), (0.12, 1.0, 0.05, 0.25), (-0.1, 1.25, -0.05, 0.2)):
            add(sphere, translate(px + ox, oy, pz + oz) @ scale(r, r, r), (0.6, 0.9, 0.55), "grass", (1, 1), 0.05, 8)


"""Для каждой из 6 ламп один раз записать, на каком расстоянии от неё находится ближайшая поверхность в любом направлении
Лампа светит во все стороны, поэтому «рисуем» сцену шесть раз — как грани куба вокруг лампы

Кубическая текстура (Cube Map) состоит из 6 квадратных граней, 
куда записываются линейные расстояния до ближайших объектов"""

# Шесть пар (направление взгляда, вектор «вверх») в том же порядке, 
# в каком OpenGL нумерует грани кубической текстуры: +X, −X, +Y, −Y, +Z, −Z.
CUBE_FACES = [((1, 0, 0), (0, -1, 0)), ((-1, 0, 0), (0, -1, 0)),
              ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, -1)),
              ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, -1, 0))]


def make_shadow_cubes(lamps, prog):
    """Для каждой лампы рендерит расстояния до ближайших объектов в 6 направлениях."""
    u = Uniforms(prog)              # Создаём кэш uniform-переменных для теневого шейдера
    fbo = glGenFramebuffers(1)      # FBO (Framebuffer Object): рисуем не в окно, а в текстуру
    rbo = glGenRenderbuffers(1)     # Создаём буфер глубины (Renderbuffer)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    # Renderbuffer размером SHADOW_RES × SHADOW_RES пикселей, глубины 24 бита, чтобы при отрисовке виден ближайший объект
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, SHADOW_RES, SHADOW_RES)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    # Прикрепляем буфер глубины к framebuffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    proj = perspective(math.pi / 2, 1.0, 0.1, SHADOW_FAR)       # Проекция с fov 90° и квадратным соотношением сторон 1.0
    # Размер области рисования — квадрат карты
    glViewport(0, 0, SHADOW_RES, SHADOW_RES)
    # Включить теневую программу
    glUseProgram(prog)
    glUniform1f(u("farPlane"), SHADOW_FAR)      # Передали дальность

    cubes = []
    for lp in lamps:
        tex = glGenTextures(1)      # Создаёт новый объект текстуры в OpenGL и получаем его числовой идентификатор
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex)     # Делаем его активным для дальнейшей настройки
        for f in range(6):
            # Выделяет память на видеокарте для всех 6 граней куба
            # Формат GL_R32F означает один 32-битный канал с плавающей точкой
            # так как здесь нужен не цвет, а одно точное число — расстояние от лампы до препятствия
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_R32F, SHADOW_RES, SHADOW_RES, 0,
                         GL_RED, GL_FLOAT, None)
        # GL_NEAREST для фильтрации: расстояния нельзя усреднять между соседними пикселями
        # ведь на границе предмета получились бы несуществующие расстояния
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        for w in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R):
            # GL_CLAMP_TO_EDGE по трём осям — не заворачивать за край
            glTexParameteri(GL_TEXTURE_CUBE_MAP, w, GL_CLAMP_TO_EDGE)
        # Передаём в шейдер мировые координаты текущей лампочки
        glUniform3f(u("lightPos"), *lp)
        for f, (d, up) in enumerate(CUBE_FACES):
            # Подключает очередную грань f кубической текстуры как цветовой выход FBO
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, tex, 0)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                sys.exit("Framebuffer для карты теней не готов (нужна поддержка R32F)")
            # залить грань единицей («расстояние = максимум», то есть ничего не мешает) и очистить глубину
            glClearColor(1, 1, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Направляет виртуальную камеру из центра лампочки lp в сторону lp + d 
            # с ориентацией up и передаёт матрицу «вид + проекция» в шейдер
            set_mat4(u("lightVP"), proj @ look_at(lp, np.asarray(lp) + d, up))
            # Перебираем все объекты сцены
            for mesh, model, *rest in objs:
                if rest[-1]:           # cast
                    # Объект отрисовывается в карту теней
                    set_mat4(u("model"), model)
                    mesh.draw()
        # Сохраняем готовую карту теней лампы в список
        cubes.append(tex)
    # Возвращаем рендеринг обратно в стандартный буфер окна (0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return cubes


# ----------------------------------------------------------------------------
# Камера
# ----------------------------------------------------------------------------
class Camera:
    def __init__(self):
        self.reset()
        self.auto = False
        self.first = True
        self.lx = self.ly = 0.0

    def reset(self):
        self.pos = np.array([0.0, 1.7, 5.0])
        self.yaw = -90.0        # смотрим вдоль −Z, к бару
        self.pitch = -6.0       # наклон вниз

    def front(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        # Вычисляем единичный 3D-вектор направления взгляда камеры по стандартной сферической формуле
        return normalize((math.cos(y) * math.cos(p), math.sin(p), math.sin(y) * math.cos(p)))

    def view(self, t):
        if self.auto:
            # автооблёт: плавное движение по залу (между столами выше мебели) и поворот головы
            x = 5.5 * math.sin(0.20 * t)                    # медленно ходим влево-вправо в пределах ±5.5 м
            z = 1.0 + 3.8 * math.sin(0.13 * t + 1.0)        # вперёд-назад от −2.8 до 4.8 м; волнисто за счёт фазы
            y = 1.7 + 0.15 * math.sin(0.6 * t)              # лёгкое «покачивание шагов»
            eye = np.array([x, y, z])
            a = 0.22 * t                   # угол, на который повёрнут взгляд
            # точка взгляда кружится вокруг камеры в горизонтальной плоскости
            target = eye + np.array([8 * math.sin(a), -0.8 + 0.5 * math.sin(0.4 * t), -8 * math.cos(a)])
            return look_at(eye, target), eye
        return look_at(self.pos, self.pos + self.front()), self.pos


cam = Camera()

# GLFW вызывает при каждом движении мыши
def on_cursor(win, x, y):
    if cam.first:
        cam.lx, cam.ly, cam.first = x, y, False
    # смещение с прошлого раза
    dx, dy = x - cam.lx, cam.ly - y
    cam.lx, cam.ly = x, y
    # в авторежиме мышь игнорируется
    if cam.auto:
        return
    # чувствительность мыши
    cam.yaw += dx * 0.1
    # ограничить наклон ±89°, чтобы избежать переворачивания камеры (эффекта gimbal lock)
    cam.pitch = max(-89.0, min(89.0, cam.pitch + dy * 0.1))


def on_key(win, key, sc, action, mods):
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(win, True)
    elif key == glfw.KEY_F:
        cam.auto = not cam.auto
    elif key == glfw.KEY_R:
        cam.reset()


def process_input(win, dt):
    if cam.auto:
        return
    speed = (6.0 if glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS else 2.5) * dt
    f = cam.front()
    r = normalize(np.cross(f, (0, 1, 0)))
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        cam.pos += f * speed
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        cam.pos -= f * speed
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        cam.pos += r * speed
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        cam.pos -= r * speed
    if glfw.get_key(win, glfw.KEY_E) == glfw.PRESS:
        cam.pos[1] += speed
    if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
        cam.pos[1] -= speed
    # камера не выходит за пределы зала
    cam.pos[0] = max(-HW + 0.3, min(HW - 0.3, cam.pos[0]))
    cam.pos[1] = max(0.3, min(ROOM_H - 0.3, cam.pos[1]))
    cam.pos[2] = max(-HD + 0.3, min(HD - 0.3, cam.pos[2]))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
    # Основную инфу брал здесь: https://github.com/totex/Learn-OpenGL-in-python/tree/master
def main():
    if not glfw.init():
        sys.exit("Не удалось инициализировать GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)                     # Задаём требуемую версию OpenGl (3.3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)     # Включаем Core Profile (современный профиль OpenGL)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)               # Включаем режим прямой совместимости (forward compatibility) для OpenGL
    glfw.window_hint(glfw.SAMPLES, 4)                                   # Включаем мультисемплинг (4x MSAA)
    # Создаём графическое окно с размерами WIDTH на HEIGHT и заголовком
    win = glfw.create_window(WIDTH, HEIGHT, "ЛР4 - Ресторан (OpenGL)", None, None)  # None-ы — не полноэкранный, без общего контекста
    # Проверяем, удалось ли создать окно
    if not win:
        glfw.terminate()
        sys.exit("Не удалось создать окно (нужен OpenGL 3.3+)")
    # Устанавливаем позицию окна
    glfw.set_window_pos(win, 600, 400)
    # Делаем созданный OpenGL-контекст окна текущим для текущего потока
    glfw.make_context_current(win)
    glfw.swap_interval(1)                                               # Включаем VSync
    glfw.set_input_mode(win, glfw.CURSOR, glfw.CURSOR_DISABLED)         # Скрываем курсор мыши и захватываем его внутри окна
    glfw.set_cursor_pos_callback(win, on_cursor)                        # Регистрируем функцию-обработчик on_cursor, которая будет вызываться каждый раз при движении мыши по окну
    glfw.set_key_callback(win, on_key)                                  # Регистрируем функцию-обработчик on_key, которая срабатывает при нажатии или отпускании клавиш на клавиатуре


    glEnable(GL_DEPTH_TEST)                         # Включаем тест глубины (Depth Testing), чтобы близкие объекты перекрывали далёкие
    glEnable(GL_MULTISAMPLE)                        # Активируем ранее запрошенное аппаратное сглаживание
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)          # Включаем бесшовное наложение текстур для кубических карт

    main_prog = build_program(MAIN_VS, MAIN_FS)     # Компилируем и связываем главную шейдерную программу из вершинного (MAIN_VS) и фрагментного (MAIN_FS) шейдеров для отрисовки самой сцены
    # Создаёт объект-обёртку, который заранее находит и кэширует адреса uniform-переменных (матриц, координат света и т. д.) внутри шейдера main_prog для быстрой передачи данных с процессора на видеокарту
    um = Uniforms(main_prog)

    # Вызывает пользовательскую функцию, которая загружает и процедурно генерирует все текстуры для сцены (материалы стен, столов, полов) и загружает их в память видеокарты
    build_textures()
    # Вызывает функцию, создающую всю 3D-геометрию ресторана (вершинные буферы VAO/VBO, мебель, декор, стены)
    build_scene()

    # Компилирует отдельную облегчённую шейдерную программу (SHADOW_VS + SHADOW_FS), нужную только для расчёта глубины (карт теней) без расчёта цветов
    shadow_prog = build_program(SHADOW_VS, SHADOW_FS)
    lamps = np.array(lamp_positions, np.float32)                # Преобразуем список координат источников света в массив NumPy
    assert len(lamps) == NUM_LIGHTS                             
    shadow_cubes = make_shadow_cubes(lamps, shadow_prog)        # Создаёт для каждой лампы кубическую карту глубины (6 направлений) и возвращает список текстур, которые будут использоваться в основном шейдере для расчёта теней
    # shadow_cubes = []

    # свет: тёплые лампы + слабый холодный свет из окон
    sun_dir = normalize((0.40, 0.65, 0.35))                     # Задаёт и нормализует вектор направления света, падающего из окон
    sun_color = np.array([0.04, 0.05, 0.09], np.float32)        # Задаёт цвет этого внешнего света в формате RGB: очень тёмный, приглушённый холодно-синий оттенок (имитация ночного неба)
    fog_lin = np.array([0.0, 0.0, 0.0])                     # Задаёт линейный цвет тумана (очень тёмный, почти чёрный оттенок) в формате RGB
    clear_col = (1.0 - np.exp(-1.1 * fog_lin)) ** (1 / 2.2)     # Вычисляет итоговый цвет очистки экрана

    last = glfw.get_time()                                      # Запоминаем текущее время
    while not glfw.window_should_close(win):
        now = glfw.get_time()
        dt, last = now - last, now
        glfw.poll_events()                       # Опрашивает очередь событий операционной системы (нажатия клавиш, движения мыши) и вызывает зарегистрированные ранее коллбэки (on_cursor, on_key)
        process_input(win, dt)                   # Обрабатывает нажатия клавиш для перемещения камеры (WASD + QE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)     # Переключаем вывод графики обратно на стандартный экранный буфер окна
        fw, fh = glfw.get_framebuffer_size(win)  # Получаем актуальные размеры буфера кадра в пикселях
        if fh == 0:
            continue
        glViewport(0, 0, fw, fh)                 # Настраиваем область отрисовки OpenGL на всю площадь окна
        glClearColor(*clear_col, 1.0)            # Заливаем экран вычисленным ранее цветом clear_col
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)          # Очищаем буфер цвета и буфер глубины от данных предыдущего кадра

        view, eye = cam.view(now)               
        proj = perspective(math.radians(65), fw / fh, 0.1, 100)     

        # Активируем главную шейдерную программу
        glUseProgram(main_prog)

        set_mat4(um("view"), view)                              # Передает в шейдер матрицы вида и проекции
        set_mat4(um("proj"), proj)

        glUniform3f(um("viewPos"), *eye)                        # Передаем координаты камеры для расчета зеркальных бликов (Specular) и тумана
        glUniform3f(um("sunDir"), *sun_dir)                     
        glUniform3f(um("sunColor"), *sun_color)                 # Передаем направление и цвет уличного света из окон
        glUniform3fv(um("pointPos"), NUM_LIGHTS, lamps)         # Передает массив координат всех ламп ресторана
        glUniform3f(um("pointColor"), 0.0, 0.0, 0.0)           # Передаем цвет ламп (тёплый жёлто-оранжевый оттенок)
        glUniform3f(um("skyAmbient"), 0.0, 0.0, 0.0)        
        glUniform3f(um("groundAmbient"), 0.0, 0.0, 0.0)     # Задаем фоновое освещение (Ambient): теплый оттенок сверху от потолка/ламп и более темный снизу от пола
        glUniform3f(um("fogColor"), *fog_lin)                   # Передаем цвет дымки
        glUniform1i(um("tex"), 0)                               # Говорит шейдеру, что основная текстура материала будет браться из 0-го текстурного слота
        glUniform1f(um("farPlane"), SHADOW_FAR)                 # Передает дальнюю границу прорисовки теней
        # Передаем в шейдер все кубические карты теней (shadow_cubes) для каждой лампы, начиная с текстурного слота 1
        for i, cube in enumerate(shadow_cubes):
            glActiveTexture(GL_TEXTURE1 + i)
            glBindTexture(GL_TEXTURE_CUBE_MAP, cube)
            glUniform1i(um(f"shadow{i}"), 1 + i)
        glActiveTexture(GL_TEXTURE0)                            # Возвращает активный слот текстур обратно на 0 перед отрисовкой обычных 2D-текстур объектов

        """Перебираем все объекты сцены (objs). 
        Каждый элемент распаковывается в 10 параметров: 
        3D-сетка (mesh), 
        матрица положения/поворота/масштаба (model), 
        базовый цвет (color), 
        ID текстуры (tex), 
        масштаб текстурных координат (uvs), 
        сила блика (spec), 
        блеск/гладкость (shin), 
        цвет собственного свечения (emis), 
        сила свечения (glow)"""
        for mesh, model, color, tex, uvs, spec, shin, emis, glow, cast in objs:
            # Передаем в шейдер индивидуальные параметры текущего объекта
            set_mat4(um("model"), model)
            glBindTexture(GL_TEXTURE_2D, tex)      
            glUniform2f(um("uvScale"), *uvs)
            glUniform3f(um("matColor"), *color)
            glUniform1f(um("matSpec"), spec)
            glUniform1f(um("matShin"), shin)
            glUniform3f(um("emissive"), *emis)
            glUniform1f(um("glow"), glow)
            mesh.draw()

        glfw.swap_buffers(win)          # показать готовый кадр (двойная буферизация: рисуем в скрытый буфер и меняем его с показанным)

    glfw.terminate()


if __name__ == "__main__":
    main()