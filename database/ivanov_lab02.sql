-- ============================================
-- СЕМЕСТРОВЫЙ ПРОЕКТ
-- ФИО: Иванов Матвей
-- ТЕМА: Разбор собственных партий в домино: где игрок теряет очки
-- ============================================


-- 1. ОПИСАНИЕ СИСТЕМЫ
-- База собственных клубных и онлайн-партий в домино (козёл) одного игрока.
-- По каждой партии хранится соперник, стартовая кость, кто ходил первым,
-- результат, формат партии, рейтинги, место игры, круг первой ошибки
-- и время на часах к этому кругу.
-- Система показывает не отдельные партии, а повторяющиеся закономерности.


-- 2. КТО ПОЛЬЗУЕТСЯ
-- Сам игрок и его тренер.


-- 3. КАКУЮ ПРОБЛЕМУ РЕШАЕТ
-- Проигранную партию можно забыть и не сделать выводы.
-- А если зацикливаться на одной партии, не видно общей картины:
-- при каких стартовых костях и при каком порядке хода результат хуже,
-- и связано ли это с рейтингом соперника или с форматом партии.


-- 4. КАКИЕ ДАННЫЕ ХРАНЯТСЯ
-- Свои клубные и онлайн-партии в домино.
-- 1. ФИО оппонента.
-- 2. Дата партии.
-- 3. Порядок хода ученика: первым или вторым.
-- 4. Результат.
-- 5. Формат партии.
-- 6. Рейтинг соперника на момент партии.
-- 7. Стартовая кость.
-- 8. Где сыграно: клуб или онлайн.
-- 9. Первая ошибка (номер круга).
-- 10. Оставшееся время к этому кругу.


-- 5. ПЯТЬ ВОПРОСОВ
-- 1. При каких стартовых костях за выбранный период результат при ходе первым хуже, чем при ходе вторым?
-- 2. Как меняется доля поражений, если соперник сильнее больше чем на 100 пунктов рейтинга?
-- 3. В каком формате (блиц, обычная, турнир) при старте с дубля-шесть процент очков меньше?
-- 4. На каких номерах кругов чаще всего происходят ошибки, сделанные менее чем с 5 минутами на часах?
-- 5. Какими стартовыми костями постоянные соперники чаще всего выигрывают у игрока?


-- ============================================
-- СТРУКТУРА БАЗЫ
-- ============================================

-- IF NOT EXISTS: если база уже есть, новая не создаётся, и повторный запуск не падает.
CREATE DATABASE IF NOT EXISTS p2_student_demo
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE p2_student_demo;

-- Таблица players: игроки, чьи партии разбираются (ученики тренера).
-- | Поле        | Что означает                 | Тип данных   |
-- | player_id   | номер игрока, первичный ключ | INTEGER      |
-- | full_name   | ФИО игрока                   | VARCHAR(120) |
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL
);

-- Таблица opponents: соперники.
-- | Поле        | Что означает                                      | Тип данных   |
-- | opponent_id | номер соперника, первичный ключ                   | INTEGER      |
-- | full_name   | ФИО соперника                                     | VARCHAR(120) |
-- | platform    | где обычно играем: club / online / app           | VARCHAR(20)  |
CREATE TABLE IF NOT EXISTS opponents (
    opponent_id INTEGER PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL,
    platform VARCHAR(20) NOT NULL
);

-- Таблица starts: справочник стартовых костей.
-- Первая выложенная кость задаёт рисунок партии так же, как дебют в шахматах.
-- | Поле       | Что означает                         | Тип данных   |
-- | start_id   | номер стартовой кости, первичный ключ | INTEGER     |
-- | tile_code  | короткий код кости                   | VARCHAR(3)   |
-- | start_name | название стартовой кости             | VARCHAR(80)  |
CREATE TABLE IF NOT EXISTS starts (
    start_id INTEGER PRIMARY KEY,
    tile_code VARCHAR(3) NOT NULL,
    start_name VARCHAR(80) NOT NULL
);

-- Таблица games: сыгранные партии, главная таблица для анализа.
-- | Поле                | Что означает                                              | Тип данных  |
-- | game_id             | номер партии, первичный ключ                              | INTEGER     |
-- | player_id           | чья партия, ссылка на players                             | INTEGER     |
-- | opponent_id         | соперник, ссылка на opponents                             | INTEGER     |
-- | start_id            | стартовая кость, ссылка на starts                         | INTEGER     |
-- | game_date           | дата партии                                               | DATE        |
-- | player_order        | порядок хода игрока: first / second                       | VARCHAR(6)  |
-- | result              | результат для игрока: win / draw / loss                   | VARCHAR(4)  |
-- | time_control        | формат: blitz / regular / tournament                      | VARCHAR(12) |
-- | player_rating       | рейтинг игрока на момент партии                           | INTEGER     |
-- | opponent_rating     | рейтинг соперника на момент партии                        | INTEGER     |
-- | venue               | где сыграно: club / online                                | VARCHAR(6)  |
-- | first_mistake_round | номер круга первой ошибки игрока (NULL, если ошибки не было) | INTEGER |
-- | clock_left_sec      | сколько секунд оставалось на часах к этому кругу          | INTEGER     |
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    player_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    start_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    player_order VARCHAR(6) NOT NULL,
    result VARCHAR(4) NOT NULL,
    time_control VARCHAR(12) NOT NULL,
    player_rating INTEGER NOT NULL,
    opponent_rating INTEGER NOT NULL,
    venue VARCHAR(6) NOT NULL,
    first_mistake_round INTEGER,
    clock_left_sec INTEGER,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (opponent_id) REFERENCES opponents(opponent_id),
    FOREIGN KEY (start_id) REFERENCES starts(start_id)
);

-- Таблица opponent_club: одна строка — одна клубная карточка соперника.
-- Связь с opponents 1:N: opponent_id здесь только внешний ключ,
-- поэтому у одного соперника может быть несколько карточек.
-- У каждой карточки ровно один соперник. Пара «соперник + клуб» не повторяется.
-- | Поле        | Что означает                                      | Тип данных   |
-- | record_id   | номер карточки, первичный ключ                    | INTEGER      |
-- | opponent_id | соперник, ссылка на opponents                     | INTEGER      |
-- | club_name   | имя клуба, уникально у одного соперника           | VARCHAR(40)  |
-- | club_rating | рейтинг по клубной классификации                  | INTEGER      |
-- | country     | страна клуба                                      | VARCHAR(40)  |
-- | city        | город клуба                                       | VARCHAR(40)  |
CREATE TABLE IF NOT EXISTS opponent_club (
    record_id INTEGER PRIMARY KEY,
    opponent_id INTEGER NOT NULL,
    club_name VARCHAR(40) NOT NULL,
    club_rating INTEGER,
    country VARCHAR(40),
    city VARCHAR(40),
    UNIQUE (opponent_id, club_name),
    FOREIGN KEY (opponent_id) REFERENCES opponents(opponent_id)
);

-- Таблица opponent_comment: комментарий игрока о сопернике.
-- Связь с opponents 1:1: opponent_id здесь одновременно PRIMARY KEY и FOREIGN KEY,
-- поэтому о сопернике не может быть двух комментариев. Комментарий есть не про всех.
-- | Поле        | Что означает                         | Тип данных   |
-- | opponent_id | соперник, ссылка на opponents        | INTEGER      |
-- | comment     | комментарий игрока об этом сопернике | VARCHAR(300) |
CREATE TABLE IF NOT EXISTS opponent_comment (
    opponent_id INTEGER PRIMARY KEY,
    comment VARCHAR(300),
    FOREIGN KEY (opponent_id) REFERENCES opponents(opponent_id)
);

-- Таблица player_opponents: с кем игрок вообще встречался.
-- Связь players–opponents M:N. Первичный ключ составной, одна и та же пара не повторяется.
-- Партии лежат отдельно в games: с одним соперником их может быть много.
-- | Поле        | Что означает                                          | Тип данных |
-- | player_id   | игрок, вместе с opponent_id входит в первичный ключ   | INTEGER    |
-- | opponent_id | соперник, вместе с player_id входит в первичный ключ  | INTEGER    |
CREATE TABLE IF NOT EXISTS player_opponents (
    player_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    PRIMARY KEY (player_id, opponent_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (opponent_id) REFERENCES opponents(opponent_id)
);


-- ============================================
-- ТЕСТОВЫЕ ДАННЫЕ
-- ============================================

INSERT IGNORE INTO players VALUES
(1, 'Иванов Матвей');

INSERT IGNORE INTO opponents VALUES
(1, 'Морозов Павел', 'club'),
(2, 'Егорова Нина', 'club'),
(3, 'Чен Артём', 'online'),
(4, 'Волков Денис', 'app'),
(5, 'Сафин Ильяс', 'online'),
(6, 'Григорьева Ольга', 'club');

INSERT IGNORE INTO starts VALUES
(1, 'D6', 'Дубль-шесть'),
(2, 'D5', 'Дубль-пять'),
(3, 'S65', 'Шесть-пять'),
(4, 'D4', 'Дубль-четыре'),
(5, 'S64', 'Шесть-четыре'),
(7, 'D1', 'Дубль-один');


INSERT IGNORE INTO games VALUES
(1,  1, 1, 1, '2026-09-02', 'first',  'loss', 'blitz',       1650, 1720, 'club',   18, 95),
(2,  1, 1, 1, '2026-09-05', 'second', 'win',  'regular',     1655, 1715, 'club',   NULL, NULL),
(3,  1, 2, 2, '2026-09-06', 'first',  'win',  'tournament',  1660, 1600, 'club',   NULL, NULL),
(4,  1, 3, 1, '2026-09-08', 'first',  'loss', 'blitz',       1662, 1810, 'online', 22, 140),
(5,  1, 4, 3, '2026-09-09', 'second', 'draw', 'regular',     1660, 1680, 'online', 31, 420),
(6,  1, 5, 4, '2026-09-11', 'second', 'loss', 'blitz',       1658, 1790, 'online', 18, 60),
(7,  1, 1, 2, '2026-09-12', 'second', 'loss', 'tournament',  1655, 1725, 'club',   27, 900),
(8,  1, 6, 5, '2026-09-14', 'first',  'win',  'regular',     1650, 1590, 'club',   NULL, NULL),
(9,  1, 3, 1, '2026-09-15', 'second', 'win',  'regular',     1655, 1800, 'online', 35, 610),
(10, 1, 2, 3, '2026-09-17', 'first',  'draw', 'tournament',  1660, 1610, 'club',   40, 1200),
(11, 1, 4, 1, '2026-09-18', 'first',  'loss', 'regular',     1658, 1690, 'online', 22, 250),
(12, 1, 5, 4, '2026-09-20', 'first',  'win',  'blitz',       1650, 1770, 'online', NULL, NULL),
(13, 1, 1, 1, '2026-09-21', 'first',  'loss', 'tournament',  1655, 1730, 'club',   29, 1500),
(14, 1, 6, 2, '2026-09-23', 'first',  'win',  'blitz',       1660, 1580, 'club',   NULL, NULL),
(15, 1, 3, 4, '2026-09-24', 'second', 'loss', 'blitz',       1662, 1820, 'online', 18, 45),
(16, 1, 2, 5, '2026-09-26', 'second', 'draw', 'regular',     1660, 1605, 'club',   33, 380),
(17, 1, 4, 3, '2026-09-27', 'first',  'win',  'blitz',       1665, 1700, 'online', NULL, NULL),
(18, 1, 5, 1, '2026-09-29', 'second', 'win',  'tournament',  1670, 1780, 'online', NULL, NULL),
(19, 1, 1, 4, '2026-10-01', 'first',  'loss', 'regular',     1668, 1735, 'club',   22, 210),
(20, 1, 3, 2, '2026-10-02', 'second', 'loss', 'regular',     1665, 1815, 'online', 25, 280),
(21, 1, 6, 1, '2026-10-04', 'second', 'win',  'blitz',       1660, 1575, 'club',   NULL, NULL),
(22, 1, 2, 2, '2026-10-05', 'first',  'win',  'regular',     1665, 1615, 'club',   NULL, NULL),
(23, 1, 4, 5, '2026-10-06', 'second', 'loss', 'blitz',       1668, 1700, 'online', 22, 150),
(24, 1, 5, 3, '2026-10-08', 'second', 'draw', 'tournament',  1665, 1785, 'online', 38, 1100),
(25, 1, 3, 7, '2026-10-09', 'first',  'win',  'blitz',       1660, 1200, 'online', 4, 170);
-- Клубные карточки вымышленные. У Морозова две карточки, в разных клубах.
-- У Чена (3), Волкова (4) и Сафина (5) карточки нет.
INSERT IGNORE INTO opponent_club VALUES
(1, 1, 'Хабаровцы', 1480, 'Россия', 'Хабаровск'),
(2, 2, 'Хабаровцы', 1390, 'Россия', 'Хабаровск'),
(3, 1, 'Морские волки', 1510, 'Россия', 'Владивосток'),
(4, 6, 'Хабаровцы', 1360, 'Россия', 'Хабаровск');

-- Комментарий есть не у всех. У Чена и Волкова записи нет.
INSERT IGNORE INTO opponent_comment VALUES
(1, 'В клубе на дубле-шесть я ему чаще проигрываю'),
(2, 'В обычном формате играет ровно, ничьих много'),
(5, 'Онлайн-соперник без клубной карточки, в турнире у меня перевес'),
(6, 'В блице против неё ошибок почти нет');

-- Пары «игрок — соперник» из уже сыгранных партий. Повтор не вставится.
INSERT IGNORE INTO player_opponents (player_id, opponent_id)
SELECT DISTINCT player_id, opponent_id
FROM games;


-- ============================================
-- ЗАПРОСЫ
-- ============================================

-- Вопрос 1: При каких стартовых костях за выбранный период результат при ходе первым хуже, чем при ходе вторым?
-- Что решает: Для каждой стартовой кости считаю процент набранных очков при ходе первым и вторым
-- (победа 1, ничья 0.5, поражение 0) за сентябрь–октябрь 2026 и оставляю кости, где первым процент ниже.
SELECT
    s.start_name,
    ROUND(AVG(CASE WHEN g.player_order = 'first'
                   THEN CASE g.result WHEN 'win' THEN 1 WHEN 'draw' THEN 0.5 ELSE 0 END
              END) * 100, 1) AS first_score_pct,
    ROUND(AVG(CASE WHEN g.player_order = 'second'
                   THEN CASE g.result WHEN 'win' THEN 1 WHEN 'draw' THEN 0.5 ELSE 0 END
              END) * 100, 1) AS second_score_pct
FROM games g
JOIN starts s ON s.start_id = g.start_id
-- WHERE g.game_date BETWEEN '2026-09-01' AND '2026-10-31'
GROUP BY s.start_name
HAVING first_score_pct < second_score_pct;

-- Вопрос 2: Как меняется доля поражений, если соперник сильнее больше чем на 100 пунктов рейтинга?
-- Что решает: Делю партии на две группы по разнице рейтингов и сравниваю в них долю поражений.
SELECT
    CASE WHEN g.opponent_rating - g.player_rating > 100
         THEN 'соперник сильнее более чем на 100'
         ELSE 'остальные партии'
    END AS rating_group,
    COUNT(*) AS games_count,
    SUM(g.result = 'loss') AS losses,
    ROUND(AVG(g.result = 'loss') * 100, 1) AS loss_pct
FROM games g
GROUP BY rating_group;

-- Вопрос 3: В каком формате (блиц, обычная, турнир) при старте с дубля-шесть процент очков меньше?
-- Что решает: Для одной стартовой кости (здесь Дубль-шесть) считаю процент очков в каждом формате,
-- первым идёт формат с худшим результатом.
SELECT
    g.time_control,
    COUNT(*) AS games_count,
    ROUND(AVG(CASE g.result WHEN 'win' THEN 1 WHEN 'draw' THEN 0.5 ELSE 0 END) * 100, 1) AS score_pct
FROM games g
JOIN starts s ON s.start_id = g.start_id
WHERE s.start_name = 'Дубль-шесть'
GROUP BY g.time_control
ORDER BY score_pct;

-- Вопрос 4: На каких номерах кругов чаще всего происходят ошибки, сделанные менее чем с 5 минутами на часах?
-- Что решает: Беру партии с первой ошибкой, сделанной при остатке меньше 300 секунд, и считаю, на каких кругах они случаются чаще.
SELECT
    g.first_mistake_round,
    COUNT(*) AS mistakes_count
FROM games g
WHERE g.first_mistake_round IS NOT NULL
  AND g.clock_left_sec < 300
GROUP BY g.first_mistake_round
ORDER BY mistakes_count DESC, g.first_mistake_round;

-- Вопрос 5: Какими стартовыми костями постоянные соперники чаще всего выигрывают у игрока?
-- Что решает: Постоянный соперник: сыграно с ним не меньше 4 партий. Считаю их победы над игроком по стартовым костям.
SELECT
    s.start_name,
    COUNT(*) AS opponent_wins
FROM games g
JOIN starts s ON s.start_id = g.start_id
WHERE g.result = 'loss'
  AND g.opponent_id IN (
      SELECT opponent_id
      FROM games
      GROUP BY opponent_id
      HAVING COUNT(*) >= 4
  )
GROUP BY s.start_name
ORDER BY opponent_wins DESC;

-- ============================================
-- 4. НОВЫЙ JOIN ДЛЯ ПРАКТИКИ № 2
-- ============================================
-- Связи в проекте:
-- 1:1  opponents - opponent_comment: у одного соперника не больше одного комментария,
--      потому что opponent_id в opponent_comment одновременно первичный и внешний ключ.
--      Комментарий есть не у всех.
-- 1:N  opponents - opponent_club: у одного соперника несколько клубных карточек,
--      у каждой карточки ровно один соперник. Пара «соперник + клуб» не повторяется.
-- 1:N  starts - games: с одной стартовой костью сыграно много партий, у каждой партии одна стартовая кость.
-- 1:N  opponents - games: с одним соперником много партий, в каждой партии один соперник.
-- 1:N  players - games: у игрока много партий, каждая партия принадлежит одному игроку.
-- M:N  players - opponents: игрок встречается со многими соперниками, соперник — со многими игроками.
--      Промежуточная таблица — player_opponents, первичный ключ составной (player_id, opponent_id),
--      поэтому одну и ту же пару второй раз записать нельзя. Партии лежат в games.

-- Вопрос: какой у меня результат против каждого соперника, в каких он клубах и что я о нём записал?
SELECT
    op.full_name AS opponent,
    c.club_name,
    c.city,
    c.club_rating,
    cm.comment,
    COUNT(g.game_id) AS games_played,
    ROUND(100 * AVG(CASE g.result WHEN 'win' THEN 1 WHEN 'draw' THEN 0.5 ELSE 0 END), 1) AS score_pct
FROM opponents op
LEFT JOIN opponent_club c ON c.opponent_id = op.opponent_id
LEFT JOIN opponent_comment cm ON cm.opponent_id = op.opponent_id
LEFT JOIN games g ON g.opponent_id = op.opponent_id
GROUP BY op.opponent_id, op.full_name, c.record_id, c.club_name, c.city, c.club_rating, cm.comment
ORDER BY c.club_rating DESC;
-- Объяснение: связал opponents и opponent_club по opponent_id через LEFT JOIN (связь 1:N),
-- opponents и opponent_comment по opponent_id через LEFT JOIN (связь 1:1),
-- opponents и games по opponent_id (связь 1:N).
-- Результат: у Морозова две клубные карточки — «Морские волки» во Владивостоке (1510)
-- и «Хабаровцы» (1480). Против него 5 партий и 20% очков, комментарий один на обе строки.
-- У Егоровой 4 партии и 75%, у Григорьевой 3 партии и 100%.
-- У Чена, Волкова и Сафина клуба нет: у Чена 40% очков и комментария нет,
-- у Волкова 37.5% и комментария нет, у Сафина 4 партии, 62.5% и комментарий есть.
-- Использую связь 1:1, потому что комментарий — отдельная запись на соперника, и она есть не у всех:
-- внутренний JOIN убрал бы Чена и Волкова.
-- Использую связь 1:N, потому что у одного соперника бывает несколько клубов,
-- и каждая карточка должна остаться отдельной строкой.

SELECT
    s.start_name,
    ROUND(AVG(CASE WHEN g.player_order = 'first'
                   THEN CASE g.result WHEN 'win' THEN 1 WHEN 'draw' THEN 0.5 ELSE 0 END
              END) * 100, 1) AS first_score_pct,
    ROUND(AVG(CASE WHEN g.player_order = 'second'
                   THEN CASE g.result WHEN 'win' THEN 1 WHEN 'draw' THEN 0.5 ELSE 0 END
              END) * 100, 1) AS second_score_pct
FROM games g
JOIN starts s ON s.start_id = g.start_id
-- WHERE g.game_date BETWEEN '2026-09-01' AND '2026-10-31'
GROUP BY s.start_name
HAVING (first_score_pct > second_score_pct OR second_score_pct = NULL);
--

-- Вопрос: у кого из соперников нет клубной карточки и оставлен ли о нём комментарий?
SELECT
    op.full_name AS opponent,
    cm.comment
FROM opponents op
LEFT JOIN opponent_club c ON c.opponent_id = op.opponent_id
LEFT JOIN opponent_comment cm ON cm.opponent_id = op.opponent_id
WHERE c.record_id IS NULL
ORDER BY op.full_name;
-- Объяснение: связал opponents с opponent_club по opponent_id через LEFT JOIN
-- и оставил строки, где карточки нет (c.record_id IS NULL).
-- Комментарий присоединил той же связью 1:1.
-- Результат: без клуба Чен Артём и Волков Денис, комментария у них тоже нет,
-- и Сафин Ильяс — клуба нет, комментарий есть.
-- Использую связь 1:N, потому что отсутствие карточки видно только если сохранить соперника,
-- даже когда справа в opponent_club пусто.
