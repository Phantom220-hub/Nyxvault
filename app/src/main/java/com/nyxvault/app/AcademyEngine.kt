package com.nyxvault.app

data class StudyLesson(
    val title: String,
    val explanation: String,
    val example: String,
    val practice: String
)

object AcademyEngine {
    private val lessons = listOf(
        StudyLesson("Python: переменные и типы", "Переменная связывает имя со значением. Основные типы: int для целых чисел, float для дробных, str для текста и bool для логических значений.", "age = 17\nname = \"Anas\"\nis_ready = True", "Создай переменные city, year и temperature и выведи их значения."),
        StudyLesson("C++: переменные и типы", "Переменная имеет тип, имя и значение. int хранит целые числа, double — числа с дробной частью, std::string — текст, bool — истину или ложь.", "int score = 100;\ndouble speed = 2.5;\nstd::string name = \"Player\";", "Объяви переменную int level со значением 1 и выведи её через std::cout."),
        StudyLesson("Алгебра: линейное уравнение", "Линейное уравнение ax + b = c решают, изолируя неизвестную. Сначала вычти b из обеих частей, затем раздели обе части на a, если a не равно нулю.", "3x + 5 = 20 → 3x = 15 → x = 5.", "Реши самостоятельно: 4x + 7 = 31."),
        StudyLesson("Физика: скорость", "Средняя скорость равна пройденному пути, делённому на время: v = s / t. В СИ путь измеряется в метрах, время — в секундах.", "Если тело прошло 100 м за 20 с, скорость равна 100 / 20 = 5 м/с.", "Какую скорость имеет велосипедист, проехавший 60 м за 12 с?"),
        StudyLesson("Химия: строение атома", "Атом состоит из ядра и электронной оболочки. В ядре находятся положительно заряженные протоны и нейтральные нейтроны; электроны имеют отрицательный заряд.", "У нейтрального атома число электронов равно числу протонов.", "Как называется число протонов в ядре атома?"),
        StudyLesson("Биология: клетка", "Клетка — базовая структурная и функциональная единица живых организмов. Мембрана отделяет её от среды, цитоплазма содержит клеточные структуры, а ДНК хранит наследственную информацию.", "Растительные клетки обычно имеют клеточную стенку, хлоропласты и крупную центральную вакуоль.", "Назови две функции клеточной мембраны."),
        StudyLesson("Godot: игровой цикл", "Godot вызывает _process(delta) каждый кадр, а _physics_process(delta) — с фиксированной частотой физики. delta помогает рассчитывать движение независимо от частоты кадров.", "position.x += speed * delta", "Почему физическое движение персонажа лучше обновлять в _physics_process()?"),
        StudyLesson("Математика: площадь прямоугольника", "Площадь прямоугольника равна произведению длины на ширину: S = a × b. Единицы площади — квадратные единицы.", "При a = 8 см и b = 3 см площадь S = 24 см².", "Найди площадь прямоугольника со сторонами 9 см и 4 см.")
    )

    fun respond(text: String): String? {
        val q = text.lowercase()
        val asksLesson = listOf("урок", "объясни тему", "научи меня", "lesson").any { q.contains(it) }
        val asksTest = listOf("тест", "проверь знания", "quiz", "практика").any { q.contains(it) }
        if (!asksLesson && !asksTest) return null
        val lesson = when {
            listOf("python", "питон").any { q.contains(it) } -> lessons[0]
            listOf("c++", "с++").any { q.contains(it) } -> lessons[1]
            listOf("алгеб", "уравнен").any { q.contains(it) } -> lessons[2]
            listOf("физик", "скорост").any { q.contains(it) } -> lessons[3]
            listOf("хими", "атом").any { q.contains(it) } -> lessons[4]
            listOf("биолог", "клет").any { q.contains(it) } -> lessons[5]
            listOf("godot", "гдот", "игров").any { q.contains(it) } -> lessons[6]
            listOf("площад", "геометр").any { q.contains(it) } -> lessons[7]
            else -> return "ACADEMY: укажи предмет и тему. Примеры: «урок по алгебре», «тест по физике», «урок Python»."
        }
        return if (asksTest) {
            "NYXVAULT ACADEMY • ПРОВЕРКА\nТема: ${lesson.title}\n\n1. ${lesson.practice}\n\nСначала ответь самостоятельно. Затем напиши «проверь мой ответ» и свой вариант — я помогу разобрать решение.\n\nКРАТКОЕ ПОВТОРЕНИЕ\n${lesson.explanation}"
        } else {
            "NYXVAULT ACADEMY • УРОК\n${lesson.title}\n\nОБЪЯСНЕНИЕ\n${lesson.explanation}\n\nПРИМЕР\n${lesson.example}\n\nПРАКТИКА\n${lesson.practice}"
        }
    }
}
