-- ============================================
-- ПРАКТИКА № 3
-- ФИО: Иванов Матвей
-- Путь Б: учебная база p3_student_demo
-- Семестровый проект (домино) не меняю: там хватает SELECT, JOIN и GROUP BY.
-- ============================================

CREATE DATABASE IF NOT EXISTS p3_student_demo
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE p3_student_demo;

-- Шаг 1. Учебные таблицы, как в разделе 0 памятки.
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS students;

CREATE TABLE students (
    id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE projects (
    id INT PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    student_id INT NOT NULL,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id)
);

INSERT INTO students (id, name) VALUES
(1, 'Аня'),
(2, 'Борис'),
(3, 'Вера');

INSERT INTO projects (id, title, student_id, created_at) VALUES
(101, 'Распознавание изображений', 1, '2026-08-20 10:00:00'),
(102, 'Учебный трекер', 1, '2026-09-03 14:30:00'),
(103, 'Система рекомендаций', 2, '2026-09-15 09:15:00');

SELECT * FROM students;
SELECT * FROM projects;


-- Шаг 2. Процедура practice_projects_since.
-- Входы: id студента и дата. Показывает имя, название проекта и дату создания.
-- Только проекты этого студента не раньше переданной даты, от ранних к поздним.
-- Если дату не передали, IF пишет «Дата не указана» и выборку не выполняет.
DROP PROCEDURE IF EXISTS practice_projects_since;
DELIMITER $$
CREATE PROCEDURE practice_projects_since(
    IN p_student_id INT,
    IN p_since DATETIME
)
BEGIN
    IF p_since IS NULL THEN
        SELECT 'Дата не указана' AS message;
    ELSE
        SELECT s.name, p.title, p.created_at
        FROM students AS s
        JOIN projects AS p ON p.student_id = s.id
        WHERE s.id = p_student_id
          AND p.created_at >= p_since
        ORDER BY p.created_at;
    END IF;
END$$
DELIMITER ;

-- Аня с 1 сентября: августовский проект отсекается, остаётся «Учебный трекер».
CALL practice_projects_since(1, '2026-09-01 00:00:00');
-- Вера с 1 сентября: проектов нет, строк 0.
CALL practice_projects_since(3, '2026-09-01 00:00:00');
-- Аня без даты: ветка IF, сообщение «Дата не указана».
CALL practice_projects_since(1, NULL);


-- Шаг 3. Функция practice_project_count_since.
-- Те же два параметра. Возвращает, сколько проектов подходит, в том числе 0.
-- Если даты нет, RETURN NULL.
DROP FUNCTION IF EXISTS practice_project_count_since;
DELIMITER $$
CREATE FUNCTION practice_project_count_since(
    p_student_id INT,
    p_since DATETIME
)
RETURNS INT
READS SQL DATA
BEGIN
    DECLARE v_count INT DEFAULT 0;
    IF p_since IS NULL THEN
        RETURN NULL;
    END IF;
    SELECT COUNT(*) INTO v_count
    FROM projects
    WHERE student_id = p_student_id
      AND created_at >= p_since;
    RETURN v_count;
END$$
DELIMITER ;

-- Аня с 1 сентября: 1. Аня с 1 октября: 0. Аня без даты: NULL.
SELECT practice_project_count_since(1, '2026-09-01 00:00:00') AS cnt;
SELECT practice_project_count_since(1, '2026-10-01 00:00:00') AS cnt;
SELECT practice_project_count_since(1, NULL) AS cnt;


-- Дополнительная проверка, дата 10 сентября 2026.
-- У Ани проекты от 20 августа и 3 сентября, оба раньше границы: 0 строк и count 0.
-- У Бориса «Система рекомендаций» от 15 сентября: 1 строка и count 1.
CALL practice_projects_since(1, '2026-09-10 00:00:00');
CALL practice_projects_since(2, '2026-09-10 00:00:00');
SELECT
    practice_project_count_since(1, '2026-09-10 00:00:00') AS anya_cnt,
    practice_project_count_since(2, '2026-09-10 00:00:00') AS boris_cnt,
    practice_project_count_since(1, '2000-09-10 00:00:00') AS full_anya_cnt;


-- Нужны ли процедура и функция в семестровом проекте.
-- Проект разбирает партии в домино и ищет, где теряются очки:
-- по стартовой кости, порядку хода, формату и сопернику.
-- Такой список уже даёт один SELECT с JOIN и WHERE,
-- а процент очков и доля поражений считаются через GROUP BY и AVG.
-- Процедура только сохранила бы тот же запрос под именем:
-- отдельных шагов вроде проверки и записи она не добавляет и быстрее запрос не делает.
-- Число партий с выбранной даты тоже считается обычным COUNT, без своей функции.
-- Поэтому в проект я их не добавляю, а создавать и вызывать показываю на учебной базе.
