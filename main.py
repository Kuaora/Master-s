import sys, pygame
import numpy as np
import math
import matplotlib.pyplot as plt
import random


WIDTH, HEIGHT = 1200, 960  # Размер окна
pygame.font.init()
font = pygame.font.SysFont('Comic Sans MS', 20)
start_ticks = pygame.time.get_ticks()


# Константы
minimum_tasks = 6
car_size_coeff = 0.8  # Коэффициент размера робота
sensor_distance = 120 * car_size_coeff  # Дальность работы лидара
car_length, car_width = 75 * car_size_coeff, 40 * car_size_coeff  # Размеры робота
max_car_speed = 80 # Максимальная скорость движения робота 50
min_distance = 50 # минимальное расстояние до цели

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)


def findIntersection(sensor, obstacles, cars, center):
    sensor_start, sensor_end = sensor.getSegment()
    intersections = []

    # Проверка пересечений с препятствиями
    for obstacle in obstacles:
        obstacle_segments = obstacle.getBB()
        for i in range(len(obstacle_segments)):
            segment_start = obstacle_segments[i - 1]
            segment_end = obstacle_segments[i]
            intersection_point = find_intersection(sensor_start, sensor_end, segment_start, segment_end)
            if intersection_point is not None:
                distance = dist(intersection_point, sensor_start)
                if distance > 0.001:
                    intersections.append(intersection_point)

    # Проверка пересечений с другими машинами
    for car in cars:
        if car == sensor.car:
            continue  # Пропускаем машину, которой принадлежит сенсор

        car_segments = car.getBB()
        for i in range(len(car_segments)):
            segment_start = car_segments[i - 1]
            segment_end = car_segments[i]
            intersection_point = find_intersection(sensor_start, sensor_end, segment_start, segment_end)
            if intersection_point is not None:
                distance = dist(intersection_point, sensor_start)
                if distance > 0.001:
                    intersections.append(intersection_point)

    center_segments = center.getBB()     # поиск пересечений с вычислительным центром
    for i in range(len(center_segments)):
        segment_start = center_segments[i - 1]
        segment_end = center_segments[i]
        intersection_point = find_intersection(sensor_start, sensor_end, segment_start, segment_end)
        if intersection_point is not None:
            distance = dist(intersection_point, sensor_start)
            if distance > 0.001:
                intersections.append(intersection_point)

    # поиск пересечений с краем карты
    screen_bb = [[+WIDTH / 2, -HEIGHT / 2],
          [+WIDTH / 2, +HEIGHT / 2],
          [-WIDTH / 2, +HEIGHT / 2],
          [-WIDTH / 2, -HEIGHT / 2]]
    screen_bb = np.add(screen_bb, [WIDTH/2, HEIGHT/2])
    for i in range(len(screen_bb)):
        segment_start = screen_bb[i - 1]
        segment_end = screen_bb[i]
        intersection_point = find_intersection(sensor_start, sensor_end, segment_start, segment_end)
        if intersection_point is not None:
            distance = dist(intersection_point, sensor_start)
            if distance > 0.001:
                intersections.append(intersection_point)

    if intersections:
        distances = [dist(p, sensor_start) for p in intersections]
        closest_index = np.argmin(distances)
        return intersections[closest_index]

    return None


def find_intersection( p0, p1, p2, p3 ):    # математика для пересечений
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]
    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0: return None # collinear
    denom_is_positive = denom > 0
    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]
    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denom_is_positive : return None # no collision
    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive : return None # no collision
    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : return None # no collision
    # collision detected
    t = t_numer / denom
    intersection_point = [ p0[0] + (t * s10_x), p0[1] + (t * s10_y) ]
    return intersection_point

# Функция поворота вектора на заданный угол
def rot(v, ang):
    s, c = math.sin(ang), math.cos(ang)
    return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]

# Функция поворота массива векторов на заданный угол
def rotArr(vv, ang):
    return [rot(v, ang) for v in vv]

# Функция вычисления расстояния между двумя точками
def dist(p1, p2):
    return np.linalg.norm(np.subtract(p1, p2))

# Функция ограничения угла в пределах [-pi, pi]
def limAng(ang):
    while ang > math.pi: ang -= 2 * math.pi
    while ang <= -math.pi: ang += 2 * math.pi
    return ang

# Функция нечеткого управления углом поворота
# Вход: dang - разница между текущим направлением и направлением на цель
# Выход: скорость поворота руля (в пределах [-1, 1])
def fuzzy_angle(dang):
    if dang < -math.pi / 2:  # Угол сильно влево
        return -1
    elif -math.pi / 2 <= dang <= math.pi / 2:  # Угол умеренный
        return dang / (math.pi / 2)
    else:  # Угол сильно вправо
        return 1

# Функция нечеткого управления скоростью
def fuzzy_speed(distance_to_goal, min_distance, max_car_speed):
    if distance_to_goal < min_distance:  # Если цель очень близко, остановиться
        return 0
    elif min_distance <= distance_to_goal <= 200:  # Умеренная скорость при средней дистанции
        return ((distance_to_goal + 25) / 225) * max_car_speed
    else:  # Максимальная скорость на большой дистанции
        return max_car_speed


def fuzzy_avoidance(Sensor_readings):
    # Применяем нечеткие правила для уворота от препятствий.
    left_obstacle = min(Sensor_readings[0:3])  # Датчики слева
    right_obstacle = min(Sensor_readings[4:7])  # Датчики справа
    center_obstacle = Sensor_readings[3]  # Центральный датчик

    # Параметры для линейных функций
    max_distance = 120  # Максимальное расстояние для нормализации
    min_distance = 40   # Минимальное расстояние для максимального поворота

    # Линейная зависимость для поворота влево в зависимости от расстояния слева
    if left_obstacle < min_distance:
        left_turn = 1  # Максимальный поворот вправо
    elif left_obstacle > max_distance:
        left_turn = 0  # Нет необходимости поворачивать вправо
    else:
        left_turn = (max_distance - left_obstacle) / (max_distance - min_distance)

    # Линейная зависимость для поворота вправо в зависимости от расстояния справа
    if right_obstacle < min_distance:
        right_turn = -1  # Максимальный поворот влево
    elif right_obstacle > max_distance:
        right_turn = 0  # Нет необходимости поворачивать влево
    else:
        right_turn = -(max_distance - right_obstacle) / (max_distance - min_distance)

    # Логика для центрального датчика
    if center_obstacle < min_distance:
        forward_turn = 0.2 if left_obstacle > right_obstacle else -0.2
    else:
        forward_turn = 0

    # Суммируем все значения для расчета поворота.
    turn = left_turn + right_turn + forward_turn

    # Ограничиваем результат в диапазоне [-1, 1] для предотвращения слишком резких поворотов
    return max(-1, min(1, turn))


def combined_steering(dang, Sensor_readings):
    # Получаем угол для движения к цели
    goal_steering = fuzzy_angle(dang)

    # Получаем угол для объезда препятствий
    obstacle_steering = fuzzy_avoidance(Sensor_readings)

    # Определяем важность каждого сигнала
    # Если препятствие близко, приоритет у объезда препятствий
    # Если нет препятствий, приоритет у движения к цели
    obstacle_distance = min(Sensor_readings)  # Минимальное расстояние до препятствия

    if obstacle_distance < 40:  # Если есть препятствия очень близко
        # Увеличиваем влияние объезда препятствий
        steering = 0.7 * obstacle_steering + 0.3 * goal_steering
    else:
        # Увеличиваем влияние движения к цели
        steering = 0.3 * obstacle_steering + 0.7 * goal_steering

    return steering

# Функция отрисовки сегмента колеса
def drawSegment(screen, p, ang, L, w):
    pp=[[p[0]-L/2, p[1]], [p[0]+L/2, p[1]]]
    c=np.mean(pp, axis=0)
    pp_=np.subtract(pp, c)
    pp=rotArr(pp_, ang)+c
    pygame.draw.line(screen, (0,0,255), *pp, w)


class Car: # Класс описания автомобиля
    def __init__(self, pos, ang, car_length, car_width, base, number):
        self.color = (0, 0, 255)
        self.number = number
        self.pos = pos  # Позиция автомобиля
        self.ang = ang  # Угол направления автомобиля
        self.L = car_length  # Длина автомобиля
        self.W = car_width  # Ширина автомобиля
        self.vLin = 0  # Линейная скорость
        self.vSteer = 0  # Скорость поворота руля
        self.angSteer = 0  # Угол поворота колес
        self.kWheels = 0.7  # Коэффициент масштаба для колес
        self.goal = None  # Цель, к которой движется автомобиль
        self.traj = []  # Траектория движения
        self.maxSteerRate = 0.05  # Максимальная скорость изменения угла поворота колес
        self.Sensor_readings = []  # Список показаний датчиков
        self.base = base  # База робота
        self.base_pos = [self.base.x, self.base.y]
        self.pause_timer = 0
        self.sens = [
            Sensor(self, self.L / 2, 0, -math.pi / 2, sensor_distance),
            Sensor(self, self.L / 2, 0, -math.pi / 3, sensor_distance),
            Sensor(self, self.L / 2, 0, -math.pi / 6, sensor_distance),
            Sensor(self, self.L / 2, 0, 0, sensor_distance),
            Sensor(self, self.L / 2, 0 / 2, math.pi / 6, sensor_distance),
            Sensor(self, self.L / 2, 0, math.pi / 3, sensor_distance),
            Sensor(self, self.L / 2, 0, math.pi / 2, sensor_distance)
        ]

    def getBB(self):
        bb = [[+self.L / 2, -self.W / 2],
              [+self.L / 2, +self.W / 2],
              [-self.L / 2, +self.W / 2],
              [-self.L / 2, -self.W / 2]]
        bb = rotArr(bb, self.ang)
        bb = np.add(bb, self.pos)
        return bb

    # Функция отрисовки автомобиля
    def draw(self, screen):
        # Построение корпуса автомобиля
        bb = [[+self.L / 2, -self.W / 2],
              [+self.L / 2, +self.W / 2],
              [-self.L / 2, +self.W / 2],
              [-self.L / 2, -self.W / 2]]
        bb = rotArr(bb, self.ang)
        bb = np.add(bb, self.pos)
        pygame.draw.polygon(screen, self.color, bb, 2)

        # Отрисовка колес
        bbWheels = (bb - np.mean(bb, axis=0)) * self.kWheels + np.mean(bb, axis=0)
        for p in bbWheels[:2]:  # Передние колеса
            drawSegment(screen, p, self.ang + self.angSteer, self.L / 5, 4)
        for p in bbWheels[2:]:  # Задние колеса
            drawSegment(screen, p, self.ang, self.L / 5, 4)

        # Отрисовка цели
        if self.goal is not None:
            pygame.draw.circle(screen, (0, 0, 0), self.goal, 4, 2)

        for sensor in self.sens:
            sensor.draw(screen)

        # Отрисовка траектории
        for i in range(1, len(self.traj)):
            pygame.draw.line(screen, (0, 200, 0), self.traj[i - 1], self.traj[i], 1)

        center_x = sum(point[0] for point in bb) / len(bb)
        center_y = sum(point[1] for point in bb) / len(bb)

        # Отображение номера в центре
        font = pygame.font.SysFont(None, 30)
        text_surface = font.render(str(self.number), True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        screen.blit(text_surface, text_rect)

    # Функция обновления состояния автомобиля
    def sim(self, dt):
        # Обновление позиции и угла направления
        vx, vy = rot((self.vLin, 0), self.ang)
        self.pos[0] += vx * dt
        self.pos[1] += vy * dt

        # Учет угла поворота колес при движении
        if self.angSteer != 0:
            R = self.L * self.kWheels / (2 * self.angSteer * dt)
            w = self.vLin / R
            self.ang += w

        # Сохранение точки траектории, если прошло значительное расстояние
        if len(self.traj) == 0 or dist(self.traj[-1], self.pos) > 10:
            self.traj.append([*self.pos])

    # Функция управления движением автомобиля
    def control(self, Sensor_readings, completed_task_counter):
        if self.goal is None:
            self.vLin = 0
            return completed_task_counter

        # Вычисление расстояния и угла до цели
        distance_to_goal = dist(self.goal, self.pos)
        vec = np.subtract(self.goal, self.pos)
        dang = limAng(math.atan2(vec[1], vec[0]) - self.ang)

        # изменение цели на базу после подбора задания
        if distance_to_goal < min_distance:
            if self.goal == self.base_pos:
                self.goal = None
                completed_task_counter += 1
            else:
                self.goal = self.base_pos
            return completed_task_counter

        # Нечеткое управление
        self.vSteer = combined_steering(dang, Sensor_readings) # Управление углом

        if min(self.Sensor_readings) < 15 or self.pause_timer > 0:
            self.vLin = 0
            self.pause_timer += 1
        else:
            self.vLin = fuzzy_speed(distance_to_goal, min_distance, max_car_speed)  # Управление скоростью

        if self.pause_timer > 60 and self.goal is not None:
            self.vLin = -max_car_speed
            self.vSteer = 1
            if min(Sensor_readings) > 60:
                self.vSteer = -1
                self.vLin = 0
                self.pause_timer = 0


        # Задержка изменения угла поворота колес
        if abs(self.angSteer - self.vSteer) > self.maxSteerRate:
            self.angSteer += self.maxSteerRate * np.sign(self.vSteer - self.angSteer)
        else:
            self.angSteer = self.vSteer

        return completed_task_counter


class Sensor:  # Класс для датчиков, установленных на роботе
    def __init__(self, car, x, y, ang, sensor_distance):
        # Инициализация датчика с его параметрами
        self.distance = sensor_distance
        self.x = x
        self.y = y
        self.ang = ang
        self.car = car

    def draw(self, screen):
        # Рисует датчик в виде линии и кружка
        p, p2 = self.getSegment()  # Получаем две точки для рисования линии
        pygame.draw.circle(screen, (255, 0, 0), p, 8 / 2, 2)
        pygame.draw.line(screen, (255, 0, 0), p, p2, 2)

    def getPos(self):
        # Метод для получения абсолютной позиции датчика в мировой системе координат
        p = rot([self.x, self.y], self.car.ang)  # Применяем вращение с учетом ориентации робота
        p = [p[0] + self.car.pos[0], p[1] + self.car.pos[1]]  # Переводим в мировые координаты робота
        return p

    def getSegment(self):
        # Метод для получения сегмента, который датчик "покрывает" в пространстве
        p = self.getPos()  # Получаем позицию датчика
        a = self.ang + self.car.ang  # Угол, под которым датчик ориентирован в мировой системе
        s, c = math.sin(a), math.cos(a)  # Вычисляем синус и косинус угла для нахождения направления
        p2 = np.add(p, [c * self.distance, s * self.distance])  # Находим координаты дальнего конца датчика
        return p, p2  # Возвращаем пару точек: начало и конец сегмента

    def getValue(self, cars):
        # Метод для получения значения датчика
        pInt = findIntersection(self, cars)  # Ищем пересечение луча датчика с препятствиями
        if pInt is None:
            return 0
        return dist(self.getPos(), pInt)  # Возвращаем расстояние до ближайшего препятствия


def drawText(screen, s, x, y):  # функция вывода текста на экран
    surf = font.render(s, True, (0, 0, 0))
    screen.blit(surf, (x, y))


class Obstacle: # Инициализация препятствия
    def __init__(self, pos, ang, number):
        self.pos = pos
        self.ang = ang * math.pi / 180
        self.L = 150
        self.W = 100
        self.number = number
        self.color = (0, 250, 255)

    def getBB(self):
        # Метод для получения ограничивающего прямоугольника (bounding box) препятствия
        # В данном случае прямоугольник определяется углами относительно центра препятствия
        bb = [[+self.L / 2, -self.W / 2],
              [+self.L / 2, +self.W / 2],
              [-self.L / 2, +self.W / 2],
              [-self.L / 2, -self.W / 2]]  # Четыре угла прямоугольника
        bb_ = np.add(rotArr(bb, self.ang), self.pos)  # Поворачиваем и сдвигаем прямоугольник в мировые координаты
        return bb_  # Возвращаем список точек прямоугольника

    def draw(self, screen):
        # Метод для отрисовки препятствия на экране
        bb_ = self.getBB()  # Получаем координаты прямоугольника
        pygame.draw.polygon(screen, self.color, bb_, 6)  # Рисуем прямоугольник с границей

        # Вычисление центра препятствия
        center_x = sum(point[0] for point in bb_) / len(bb_)
        center_y = sum(point[1] for point in bb_) / len(bb_)

        # Отображение номера в центре
        font = pygame.font.SysFont(None, 30)
        text_surface = font.render(str(self.number), True, self.color)
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        screen.blit(text_surface, text_rect)


class TaskSystem:
    POSSIBLE_TASKS = [
        (173, 303), (68, 150), (275, 25), (431, 170),
        (677, 180), (923, 48), (1146, 244), (845, 240),
        (770, 466), (1115, 481), (761, 818), (521, 859),
        (113, 828), (240, 561), (856, 553), (261, 451)
    ]

    def __init__(self, cars, bases, method='fuzzy', max_tasks_per_car=1):
        self.cars = cars
        self.bases = bases
        self.method = method  # 'fuzzy' или 'greedy'
        self.task_radius = 5
        self.max_tasks_per_car = max_tasks_per_car
        self.task_lifetime_limit = 30  # Время (сек), через которое задача становится полностью красной
        self.completed_task_durations = []  # Время жизни выполненных задач
        self.active_tasks = self._initialize_tasks()
        self.car_tasks = {car: [] for car in self.cars}

    def _initialize_tasks(self):
        return [{"position": pos, "created_at": 0, "assigned": None}
                for pos in random.sample(self.POSSIBLE_TASKS, minimum_tasks)]

    def draw_tasks(self, screen, seconds):
        for task in self.active_tasks:
            age = seconds - task["created_at"]
            alpha = min(age / self.task_lifetime_limit, 1.0)  # 0 — зелёный, 1 — красный

            r = int(255 * alpha)
            g = int(255 * (1 - alpha))
            color = (r, g, 0)

            pygame.draw.circle(screen, color, task["position"], self.task_radius)

    def pick_up_task(self, car, goal, seconds):
        if self.car_tasks[car]:
            task = self.car_tasks[car][0]
            if goal == task["position"] and math.dist(car.pos, task["position"]) <= min_distance:
                lifetime = seconds - task["created_at"]
                self.completed_task_durations.append(lifetime)

                self.car_tasks[car].pop(0)
                self.active_tasks.remove(task)
                self._regenerate_task(seconds)
                car.goal = car.base_pos  # Возврат на базу

    def _regenerate_task(self, seconds):
        if len(self.active_tasks) < minimum_tasks:
            active_positions = {tuple(task["position"]) for task in self.active_tasks}
            available = list(set(self.POSSIBLE_TASKS) - active_positions)
            if available:
                self.active_tasks.append({
                    "position": random.choice(available),
                    "created_at": seconds,
                    "assigned": None
                })

    def update_tasks(self, seconds):
        for task in self.active_tasks:
            if task["assigned"] is None:
                car = self._find_best_car(task, seconds)
                if car and len(self.car_tasks[car]) < self.max_tasks_per_car:
                    task["assigned"] = car
                    self.car_tasks[car].append(task)

        for car in self.cars:
            self._sort_tasks(car, seconds)

        for car in self.cars:
            if not car.goal and self.car_tasks[car]:
                if dist(car.pos, car.base_pos) <= min_distance:
                    car.goal = self.car_tasks[car][0]["position"]

    def _find_best_car(self, task, seconds):
        eligible_cars = [car for car in self.cars if len(self.car_tasks[car]) < self.max_tasks_per_car]
        if self.method == 'greedy':
            return min(eligible_cars,
                       key=lambda car: dist(car.base_pos, task["position"]),
                       default=None)
        elif self.method == 'fuzzy':
            return min(eligible_cars,
                       key=lambda car: (
                           0.5 * np.exp(-(seconds - task["created_at"]) / 30) +
                           0.3 * dist(car.base_pos, task["position"]) / (dist(car.base_pos, task["position"]) + 100) +
                           0.2 * self._fuzzy_load(car) ** 2
                       ),
                       default=None)

    def _fuzzy_load(self, car):
        tasks = self.car_tasks[car]
        count = len(tasks)
        distance_sum = sum(dist(t["position"], car.base_pos) for t in tasks)
        return min(1, (count + distance_sum / 5))

    def _sort_tasks(self, car, seconds):
        if self.method == 'greedy':
            self.car_tasks[car].sort(key=lambda t: dist(car.base_pos, t["position"]))
        elif self.method == 'fuzzy':
            self.car_tasks[car].sort(
                key=lambda task: (
                    0.5 * np.exp(-(seconds - task["created_at"]) / 30) +
                    0.3 * dist(car.base_pos, task["position"]) / (dist(car.base_pos, task["position"]) + 100) +
                    0.2 * self._fuzzy_load(car) ** 2

                )
            )

    def get_average_task_time(self):
        if not self.completed_task_durations:
            return 0
        return sum(self.completed_task_durations) / len(self.completed_task_durations)



class Base:
    def __init__(self, x, y, radius=20):
        self.x = x
        self.y = y
        self.pos = [x,y]
        self.radius = radius
        self.color = (0, 0, 255)
        self.thickness = 3

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius, self.thickness)


class Center: # Инициализация центра
    def __init__(self, pos, ang):
        self.pos = pos
        self.ang = ang * math.pi / 180
        self.L = 60
        self.W = 60

    def getBB(self):
        # Метод для получения ограничивающего прямоугольника (bounding box) препятствия
        # В данном случае прямоугольник определяется углами относительно центра препятствия
        bb = [[+self.L / 2, -self.W / 2],
              [+self.L / 2, +self.W / 2],
              [-self.L / 2, +self.W / 2],
              [-self.L / 2, -self.W / 2]]  # Четыре угла прямоугольника
        bb_ = np.add(rotArr(bb, self.ang), self.pos)  # Поворачиваем и сдвигаем прямоугольник в мировые координаты
        return bb_  # Возвращаем список точек прямоугольника

    def draw(self, screen):
        # Метод для отрисовки препятствия на экране
        bb_ = self.getBB()  # Получаем координаты прямоугольника
        pygame.draw.polygon(screen, (0, 250, 0), bb_, 6)  # Рисуем прямоугольник с границей


def plot_fuzzy_systems():
    # График для управления углом
    angles = np.linspace(-math.pi, math.pi, 100)
    angle_controls = [fuzzy_angle(a) for a in angles]

    # График для управления скоростью
    distances = np.linspace(0, 300, 100)
    distance_controls = [fuzzy_speed(distance, min_distance, max_car_speed) for distance in distances]

    # График для объезда препятствий
    sensor_angles = [-math.pi / 2, -math.pi / 3, -math.pi / 6, 0, math.pi / 6, math.pi / 3, math.pi / 2]
    sensor_readings = [120] * 7  # Все датчики "видят" большое расстояние

    obstacle_positions = np.linspace(-math.pi / 2, math.pi / 2, 100)  # Углы препятствий
    avoidance_controls = []

    for obstacle_angle in obstacle_positions:
        # Моделируем активацию одного датчика на основе положения препятствия
        sensor_readings = [
            max(120 - abs(obstacle_angle - sensor_angle) * 240 / math.pi, 0)
            for sensor_angle in sensor_angles
        ]
        avoidance_controls.append(fuzzy_avoidance(sensor_readings))

    # Построение графиков
    plt.figure(figsize=(15, 5))

    # Подграфик для угла
    plt.subplot(1, 3, 1)
    plt.plot(angles, angle_controls, label='Управление углом')
    plt.title('Нечеткая логика угла')
    plt.xlabel('Δ угол (радианы)')
    plt.ylabel('Скорость поворота руля')
    plt.grid()
    plt.legend()

    # Подграфик для скорости
    plt.subplot(1, 3, 2)
    plt.plot(distances, distance_controls, label='Управление скоростью', color='orange')
    plt.title('Нечеткая логика скорости')
    plt.xlabel('Расстояние до цели (пиксели)')
    plt.ylabel('Скорость движения')
    plt.grid()
    plt.legend()

    # Подграфик для объезда препятствий
    plt.subplot(1, 3, 3)
    plt.plot(obstacle_positions, avoidance_controls, label='Управление объездом', color='green')
    plt.title('Реакция на препятствия')
    plt.xlabel('Направление до препятствия (радианы)')
    plt.ylabel('Угол поворота руля для объезда')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    def nonlinear_metric(t, d, l):
        # Нелинейная функция метрики
        time_component = 0.5 * np.exp(-t / 30)
        distance_component = 0.3 * (d / (d + 100))
        load_component = 0.2 * l ** 2
        return time_component + distance_component + load_component

    def plot_nonlinear_metric():
        # Диапазоны
        times = np.linspace(0, 90, 100)
        distances = np.linspace(0, 500, 100)
        loads = np.linspace(0, 1, 100)

        # Фиксированные значения
        fixed_d = 250
        fixed_t = 45
        fixed_l = 0.5

        # Метрики
        score_by_time = nonlinear_metric(times, fixed_d, fixed_l)
        score_by_distance = nonlinear_metric(fixed_t, distances, fixed_l)
        score_by_load = nonlinear_metric(fixed_t, fixed_d, loads)

        # Построение графиков
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(times, score_by_time, label='Метрика по времени')
        plt.xlabel('Время задачи (с)')
        plt.ylabel('Метрика')
        plt.title('Метрика по времени')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(distances, score_by_distance, label='Метрика по расстоянию', color='orange')
        plt.xlabel('Расстояние до задачи (px)')
        plt.ylabel('Метрика')
        plt.title('Метрика по расстоянию')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(loads, score_by_load, label='Метрика по загрузке', color='green')
        plt.xlabel('Загрузка (0–1)')
        plt.ylabel('Метрика')
        plt.title('Метрика по загрузке')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    plot_nonlinear_metric()



# Основная функция программы
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    timer = pygame.time.Clock()

    Bases = [
        Base(WIDTH/3, HEIGHT/3),
        Base(WIDTH/3*2, HEIGHT/3),
        Base(WIDTH/2, HEIGHT/3*2)
    ]

    # Создание автомобилей
    Cars = [
        Car([Bases[0].x, Bases[0].y], 1.5, car_length, car_width, Bases[0], 1),
        Car([Bases[1].x, Bases[1].y], 1.5, car_length, car_width, Bases[1], 2),
        Car([Bases[2].x, Bases[2].y], 1.5, car_length, car_width, Bases[2], 3)
    ]

    center = Center([WIDTH/2 - 30, HEIGHT/2 - 20], 0)

    # Выбор алгоритма распределения задач
    #method_choice = input("Выберите алгоритм распределения задач:\n1 - Нечеткая логика\n2 - Жадный (по расстоянию)\nВаш выбор: ")
    method_choice = "1"
    if method_choice == "1":
        method = "fuzzy"
    elif method_choice == "2":
        method = "greedy"
    else:
        print("Неверный выбор, используется алгоритм по умолчанию (нечеткая логика).")
        method = "fuzzy"

    tasks = TaskSystem(Cars, Bases, method)
    completed_task_counter = 0


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     x, y = event.pos  # Установка новой цели по клику мыши
            #     goal = [x, y]
            if event.type == pygame.KEYDOWN:
                # checking if key "A" was pressed
                if event.key == pygame.K_a:
                    plot_fuzzy_systems()  # Построение графиков перед запуском симуляции

        screen.fill((255, 255, 255))  # Очистка экрана
        seconds = (pygame.time.get_ticks() - start_ticks) / 1000


        obstacles = [   # создание препятствий
            Obstacle([750, 50], 0, 1),
            Obstacle([270, 200], 55, 2),
            Obstacle([100, 450], 90, 3),
            Obstacle([250, 750], 75, 4),
            Obstacle([950, 770], 135, 5),
            Obstacle([1000, 270], 90, 6)]
        for obstacle in obstacles:
            obstacle.draw(screen)

        tasks.update_tasks(seconds)
        tasks.draw_tasks(screen, seconds)

        for car in Cars:
            # поиск пересечений сенсоров с препятствиями
            car.Sensor_readings = []
            for sensor in car.sens:
                obstcl_coordinate = findIntersection(sensor, obstacles, Cars, center)
                if obstcl_coordinate is not None:
                    distance = dist(obstcl_coordinate, Sensor.getPos(car.sens[0]))
                    car.Sensor_readings.append(distance)
                else: # если нет препятствия, добавляется макс дистанция сенсора
                    car.Sensor_readings.append(120)

            # Назначение цели из списка задач
            if not car.goal and tasks.car_tasks[car]:
                car.goal = tasks.car_tasks[car][0]['position']

            tasks.pick_up_task(car, car.goal, seconds)  # Проверка на захват задачи
            completed_task_counter = car.control(car.Sensor_readings, completed_task_counter)


        center.draw(screen)

        for base in Bases: # Отрисовка баз роботов
            base.draw(screen)

        for car in Cars:
            car.sim(1 / 60)  # Обновление состояния автомобиля
            car.draw(screen)  # Отрисовка автомобиля

        Average_task_life = tasks.get_average_task_time()

        drawText(screen, f"time = {seconds:.2f}", 350, 5)  # время симуляции
        drawText(screen, f"completed tasks = {int(completed_task_counter)}", 350, 25)  # счетчик выполненных задач
        drawText(screen, f"current method = {method}", 20, 5)  # счетчик выполненных задач
        drawText(screen, f"Average task life = {int(Average_task_life)}", 20, 25)  # счетчик cреднего времени сущ задач

        pygame.display.flip()  # Обновление экрана
        timer.tick(60)  # Установка частоты обновления

if __name__ == "__main__":
    #plot_fuzzy_systems()  # Построение графиков перед запуском симуляции
    main()
