-- Учебная база «Аня, Борис, Вера» · SQLite
-- Скопируйте ВЕСЬ текст в SQLiteOnline и нажмите Run.
-- Если запускаете в старой вкладке, данные будут удалены и созданы заново.
PRAGMA foreign_keys = ON;
DROP TABLE IF EXISTS enrollments;
DROP TABLE IF EXISTS student_cards;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS courses;
DROP TABLE IF EXISTS students;

-- СТУДЕНТЫ
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
INSERT INTO students (id, name) VALUES
(1, 'Аня'), (2, 'Борис'), (3, 'Вера');

-- ПРОЕКТЫ: у одного студента может быть несколько проектов (1:N)
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    student_id INTEGER NOT NULL REFERENCES students(id)
);
INSERT INTO projects (id, title, student_id) VALUES
(101, 'Магазин', 1),
(102, 'Бот', 1),
(103, 'Игра', 2);

-- СТУДЕНЧЕСКИЕ КАРТОЧКИ: у одного студента не более одной (1:1)
CREATE TABLE student_cards (
    student_id INTEGER PRIMARY KEY REFERENCES students(id),
    card_number TEXT NOT NULL UNIQUE
);
INSERT INTO student_cards (student_id, card_number) VALUES
(1, 'CARD-001'), (2, 'CARD-002');

-- КУРСЫ
CREATE TABLE courses (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL UNIQUE
);
INSERT INTO courses (id, title) VALUES
(10, 'SQL'), (20, 'Python'), (30, 'Алгоритмы');

-- ЗАПИСИ НА КУРСЫ: один студент — много курсов; на одном курсе — много студентов (M:N)
-- Поэтому первичный ключ здесь СОСТАВНОЙ: (student_id, course_id)
CREATE TABLE enrollments (
    student_id INTEGER NOT NULL REFERENCES students(id),
    course_id INTEGER NOT NULL REFERENCES courses(id),
    PRIMARY KEY (student_id, course_id)
);
INSERT INTO enrollments (student_id, course_id) VALUES
(1, 10), (1, 20), (2, 10);

-- ПРОВЕРКА: должно быть 3 студента, 3 проекта, 2 карточки,
-- 3 курса, 3 записи на курсы.
SELECT 'students' AS table_name, COUNT(*) AS rows_count FROM students
UNION ALL SELECT 'projects', COUNT(*) FROM projects
UNION ALL SELECT 'student_cards', COUNT(*) FROM student_cards
UNION ALL SELECT 'courses', COUNT(*) FROM courses
UNION ALL SELECT 'enrollments', COUNT(*) FROM enrollments;
