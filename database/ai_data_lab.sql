-- AI & Data Analytics Lab: учебная база для лабораторных работ по SQL
-- Скрипт использует базовые конструкции, совместимые с MySQL 8+ и PostgreSQL.
-- Запускайте его ВНУТРИ отдельной пустой базы данных ai_data_lab.
-- ВНИМАНИЕ: скрипт удаляет одноименные учебные таблицы, если они уже существуют.

DROP VIEW IF EXISTS run_overview;
DROP TABLE IF EXISTS runs;
DROP TABLE IF EXISTS experiments;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS datasets;
DROP TABLE IF EXISTS researchers;

CREATE TABLE researchers (
    id INTEGER PRIMARY KEY,
    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(160) NOT NULL,
    team VARCHAR(30) NOT NULL
);

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    task_type VARCHAR(30) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    records_count INTEGER NOT NULL
);

CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    family VARCHAR(80) NOT NULL,
    task_type VARCHAR(30) NOT NULL,
    parameters_mln DECIMAL(10,2) NOT NULL
);

CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    title VARCHAR(140) NOT NULL,
    researcher_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (researcher_id) REFERENCES researchers(id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL,
    gpu VARCHAR(30) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    duration_min INTEGER NOT NULL,
    learning_rate DECIMAL(10,6) NOT NULL,
    batch_size INTEGER NOT NULL,
    accuracy DECIMAL(6,4),
    f1_score DECIMAL(6,4),
    notes VARCHAR(120),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

INSERT INTO researchers (id, full_name, email, team) VALUES
(1, 'Анна Волкова', 'anna.volkova@ai-lab.local', 'CV'),
(2, 'Иван Ким', 'ivan.kim@ai-lab.local', 'NLP'),
(3, 'Мария Орлова', 'maria.orlova@ai-lab.local', 'Tabular'),
(4, 'Алексей Пак', 'alexey.pak@ai-lab.local', 'CV'),
(5, 'Елена Соколова', 'elena.sokolova@ai-lab.local', 'NLP'),
(6, 'Даниил Ли', 'daniil.li@ai-lab.local', 'Tabular'),
(7, 'Софья Миронова', 'sofia.mironova@ai-lab.local', 'CV'),
(8, 'Максим Чернов', 'maxim.chernov@ai-lab.local', 'NLP'),
(9, 'Полина Кравцова', 'polina.kravtsova@ai-lab.local', 'Tabular'),
(10, 'Артём Юн', 'artem.yun@ai-lab.local', 'CV'),
(11, 'Ольга Белова', 'olga.belova@ai-lab.local', 'NLP'),
(12, 'Никита Фролов', 'nikita.frolov@ai-lab.local', 'Tabular'),
(13, 'Дарья Новикова', 'daria.novikova@ai-lab.local', 'CV'),
(14, 'Роман Козлов', 'roman.kozlov@ai-lab.local', 'NLP');

INSERT INTO datasets (id, name, task_type, source_type, records_count) VALUES
(1, 'CIFAR-10', 'CV', 'public', 60000),
(2, 'Fashion-MNIST', 'CV', 'public', 70000),
(3, 'Cats vs Dogs', 'CV', 'public', 25000),
(4, 'PlantVillage', 'CV', 'public', 54303),
(5, 'IMDB Reviews', 'NLP', 'public', 50000),
(6, 'AG News', 'NLP', 'public', 120000),
(7, 'RuSentiment', 'NLP', 'internal', 31000),
(8, 'Support Tickets RU', 'NLP', 'internal', 82500),
(9, 'Customer Churn', 'Tabular', 'internal', 10000),
(10, 'Credit Default', 'Tabular', 'public', 30000),
(11, 'Fraud Mini', 'Tabular', 'internal', 125000),
(12, 'Student Success', 'Tabular', 'internal', 18000);

INSERT INTO models (id, name, family, task_type, parameters_mln) VALUES
(1, 'ResNet50', 'ResNet', 'CV', 25.6),
(2, 'EfficientNet-B0', 'EfficientNet', 'CV', 5.3),
(3, 'ViT-B16', 'Vision Transformer', 'CV', 86.6),
(4, 'MobileNetV3', 'MobileNet', 'CV', 5.4),
(5, 'BERT-base', 'BERT', 'NLP', 110.0),
(6, 'DistilBERT', 'BERT', 'NLP', 66.0),
(7, 'RoBERTa-base', 'RoBERTa', 'NLP', 125.0),
(8, 'RuBERT-tiny2', 'BERT', 'NLP', 29.4),
(9, 'XGBoost', 'Gradient Boosting', 'Tabular', 2.1),
(10, 'LightGBM', 'Gradient Boosting', 'Tabular', 1.8),
(11, 'RandomForest', 'Tree Ensemble', 'Tabular', 0.9),
(12, 'CatBoost', 'Gradient Boosting', 'Tabular', 2.5),
(13, 'ConvNeXt-Tiny', 'ConvNeXt', 'CV', 28.6),
(14, 'DeBERTa-v3-small', 'DeBERTa', 'NLP', 44.0),
(15, 'LogisticRegression', 'Linear', 'Tabular', 0.05);

INSERT INTO experiments (id, title, researcher_id, dataset_id, model_id, created_at) VALUES
(1, 'nlp_tuned_001', 8, 7, 6, '2026-08-28 15:00:00'),
(2, 'cv_sanity_check_002', 10, 3, 13, '2026-08-31 10:00:00'),
(3, 'tabular_class_weight_003', 3, 11, 11, '2026-09-10 16:00:00'),
(4, 'cv_new_features_004', 7, 3, 4, '2026-09-01 14:00:00'),
(5, 'tabular_baseline_005', 12, 9, 10, '2026-08-28 11:00:00'),
(6, 'nlp_dropout_006', 8, 8, 14, '2026-09-04 09:00:00'),
(7, 'nlp_augmentation_007', 11, 5, 14, '2026-08-27 12:00:00'),
(8, 'cv_tuned_008', 13, 1, 1, '2026-08-28 11:00:00'),
(9, 'cv_new_features_009', 1, 2, 1, '2026-08-29 14:00:00'),
(10, 'nlp_new_features_010', 8, 8, 8, '2026-09-17 11:00:00'),
(11, 'nlp_augmentation_011', 2, 8, 5, '2026-08-25 12:00:00'),
(12, 'nlp_class_weight_012', 11, 8, 7, '2026-08-22 14:00:00'),
(13, 'cv_tuned_013', 7, 2, 1, '2026-09-15 14:00:00'),
(14, 'cv_augmentation_014', 4, 2, 2, '2026-09-15 16:00:00'),
(15, 'cv_sanity_check_015', 7, 2, 2, '2026-08-25 15:00:00'),
(16, 'nlp_dropout_016', 2, 5, 6, '2026-08-27 09:00:00'),
(17, 'tabular_baseline_017', 9, 11, 12, '2026-09-01 15:00:00'),
(18, 'cv_new_features_018', 4, 1, 2, '2026-08-21 14:00:00'),
(19, 'tabular_augmentation_019', 3, 12, 9, '2026-09-11 10:00:00'),
(20, 'nlp_new_features_020', 8, 8, 6, '2026-09-24 09:00:00'),
(21, 'tabular_tuned_021', 6, 9, 11, '2026-09-12 14:00:00'),
(22, 'nlp_class_weight_022', 2, 7, 14, '2026-09-17 11:00:00'),
(23, 'cv_dropout_023', 10, 4, 1, '2026-09-04 12:00:00'),
(24, 'tabular_tuned_024', 3, 12, 11, '2026-08-26 15:00:00'),
(25, 'tabular_lr_sweep_025', 12, 11, 10, '2026-09-21 10:00:00');

INSERT INTO experiments (id, title, researcher_id, dataset_id, model_id, created_at) VALUES
(26, 'tabular_dropout_026', 12, 9, 10, '2026-09-16 11:00:00'),
(27, 'cv_baseline_027', 1, 2, 1, '2026-08-26 13:00:00'),
(28, 'cv_lr_sweep_028', 4, 4, 1, '2026-09-08 14:00:00'),
(29, 'nlp_lr_sweep_029', 2, 5, 6, '2026-09-14 13:00:00'),
(30, 'tabular_dropout_030', 12, 12, 9, '2026-09-21 12:00:00'),
(31, 'cv_dropout_031', 1, 2, 1, '2026-09-23 14:00:00'),
(32, 'cv_baseline_032', 4, 1, 2, '2026-09-05 14:00:00'),
(33, 'tabular_lr_sweep_033', 6, 10, 15, '2026-09-11 13:00:00'),
(34, 'nlp_sanity_check_034', 8, 6, 5, '2026-09-02 09:00:00'),
(35, 'tabular_dropout_035', 12, 11, 10, '2026-09-08 13:00:00'),
(36, 'tabular_sanity_check_036', 3, 11, 15, '2026-09-22 10:00:00'),
(37, 'nlp_new_features_037', 14, 6, 14, '2026-08-20 10:00:00'),
(38, 'tabular_baseline_038', 6, 10, 10, '2026-09-21 14:00:00'),
(39, 'cv_tuned_039', 1, 3, 1, '2026-09-20 11:00:00'),
(40, 'tabular_new_features_040', 12, 9, 11, '2026-08-30 11:00:00'),
(41, 'tabular_lr_sweep_041', 9, 11, 10, '2026-09-11 14:00:00'),
(42, 'tabular_sanity_check_042', 3, 10, 15, '2026-09-11 16:00:00'),
(43, 'cv_baseline_043', 1, 3, 4, '2026-08-23 15:00:00'),
(44, 'tabular_dropout_044', 6, 12, 12, '2026-09-13 12:00:00'),
(45, 'nlp_tuned_045', 2, 6, 5, '2026-09-22 10:00:00');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(1, 1, 'completed', 'RTX 4090', '2026-08-31 15:51:00', 38, 0.003, 64, 0.8618, 0.8688, 'debug'),
(2, 2, 'completed', 'H100', '2026-09-01 10:05:00', 183, 0.0003, 32, 0.7634, 0.7586, 'fast_test'),
(3, 3, 'completed', 'RTX 4090', '2026-09-13 23:14:00', 191, 0.0001, 128, 0.8547, 0.8013, 'full_train'),
(4, 4, 'completed', 'T4', '2026-09-04 08:19:00', 127, 0.0003, 64, 0.8055, 0.8287, 'full_train'),
(5, 5, 'completed', 'A100', '2026-08-28 16:46:00', 129, 0.0001, 64, 0.9473, 0.9254, 'full_train'),
(6, 6, 'failed', 'T4', '2026-09-04 21:23:00', 17, 0.0001, 32, 0.6171, NULL, 'production_candidate'),
(7, 7, 'completed', 'L40S', '2026-08-28 09:10:00', 195, 0.001, 32, 0.9198, 0.8749, 'debug'),
(8, 8, 'completed', 'CPU', '2026-09-01 11:02:00', 165, 5e-05, 32, 0.7537, 0.7308, 'class_weight'),
(9, 9, 'completed', 'CPU', '2026-09-01 09:38:00', 66, 0.01, 32, 0.8899, 0.9042, 'fast_test'),
(10, 10, 'failed', 'H100', '2026-09-18 07:21:00', 225, 0.0001, 64, 0.5504, NULL, 'augmentation'),
(11, 11, 'failed', 'H100', '2026-08-28 01:44:00', 199, 5e-05, 64, NULL, NULL, 'baseline'),
(12, 12, 'completed', 'A100', '2026-08-25 02:29:00', 112, 0.0001, 16, 0.8816, 0.9066, 'fast_test'),
(13, 13, 'completed', 'A100', '2026-09-16 09:13:00', 67, 0.0001, 16, 0.9406, 0.9577, 'tuned'),
(14, 14, 'completed', 'L40S', '2026-09-17 17:04:00', 233, 0.0001, 16, 0.8477, 0.85, 'debug'),
(15, 15, 'completed', 'T4', '2026-08-26 16:04:00', 40, 0.003, 64, 0.7947, 0.7395, 'fast_test'),
(16, 16, 'completed', 'T4', '2026-08-30 09:12:00', 184, 0.001, 64, 0.985, 0.98, 'full_train'),
(17, 17, 'completed', 'A100', '2026-09-04 08:35:00', 59, 0.01, 128, 0.7296, 0.7422, 'class_weight'),
(18, 18, 'completed', 'RTX 4090', '2026-08-24 14:27:00', 69, 0.01, 64, 0.867, 0.8915, 'tuned'),
(19, 19, 'cancelled', 'L40S', '2026-09-11 21:23:00', 247, 0.001, 128, NULL, NULL, 'production_candidate'),
(20, 20, 'completed', 'CPU', '2026-09-28 09:32:00', 241, 0.01, 64, 0.934, 0.8968, 'production_candidate'),
(21, 21, 'completed', 'CPU', '2026-09-13 08:48:00', 94, 0.01, 16, 0.8026, 0.7907, 'class_weight'),
(22, 22, 'completed', 'T4', '2026-09-18 18:51:00', 130, 0.0001, 16, 0.9291, 0.9301, 'tuned'),
(23, 23, 'cancelled', 'H100', '2026-09-06 06:09:00', 108, 0.003, 64, NULL, NULL, 'augmentation'),
(24, 24, 'completed', 'L40S', '2026-08-29 16:19:00', 264, 0.0003, 128, 0.881, 0.8698, 'baseline'),
(25, 25, 'completed', 'A100', '2026-09-10 11:00:00', 198, 0.001, 64, 0.721, 0.694, 'baseline');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(26, 26, 'completed', 'A100', '2026-09-19 22:02:00', 189, 0.0001, 32, 0.8745, 0.8376, 'augmentation'),
(27, 27, 'completed', 'RTX 4090', '2026-08-29 15:47:00', 135, 0.003, 32, 0.9226, 0.8859, 'production_candidate'),
(28, 28, 'completed', 'T4', '2026-09-09 07:55:00', 276, 0.0003, 64, 0.9403, 0.9514, 'full_train'),
(29, 29, 'cancelled', 'RTX 4090', '2026-09-17 10:32:00', 60, 0.001, 128, 0.6162, NULL, 'augmentation'),
(30, 30, 'completed', 'RTX 4090', '2026-09-24 13:50:00', 186, 5e-05, 128, 0.7616, 0.7043, 'baseline'),
(31, 31, 'failed', 'CPU', '2026-09-24 19:57:00', 207, 0.0001, 32, NULL, NULL, 'production_candidate'),
(32, 32, 'completed', 'H100', '2026-09-07 09:47:00', 127, 5e-05, 16, 0.84, 0.8091, 'augmentation'),
(33, 33, 'completed', 'A100', '2026-09-14 10:14:00', 60, 5e-05, 32, 0.8462, 0.8282, 'full_train'),
(34, 34, 'failed', 'H100', '2026-09-03 13:27:00', 187, 0.0003, 16, NULL, NULL, 'production_candidate'),
(35, 35, 'completed', 'A100', '2026-09-10 07:48:00', 204, 0.001, 32, 0.8785, 0.8335, 'production_candidate'),
(36, 36, 'failed', 'A100', '2026-09-24 09:17:00', 77, 0.01, 16, NULL, NULL, 'class_weight'),
(37, 37, 'cancelled', 'A100', '2026-08-22 23:02:00', 101, 0.0003, 16, NULL, 0.6992, 'full_train'),
(38, 38, 'completed', 'H100', '2026-09-22 05:30:00', 97, 5e-05, 16, 0.8609, 0.84, 'fast_test'),
(39, 39, 'completed', 'T4', '2026-09-23 17:21:00', 149, 0.01, 32, 0.8938, 0.9127, 'fast_test'),
(40, 40, 'completed', 'H100', '2026-08-31 20:26:00', 40, 0.01, 128, 0.8067, 0.7747, 'augmentation'),
(41, 41, 'completed', 'T4', '2026-09-12 14:38:00', 25, 0.001, 128, 0.8415, 0.796, 'augmentation'),
(42, 42, 'failed', 'T4', '2026-09-15 03:32:00', 226, 5e-05, 32, 0.5982, 0.6939, 'class_weight'),
(43, 43, 'completed', 'A100', '2026-08-24 07:34:00', 208, 0.0001, 64, 0.8797, 0.8213, 'class_weight'),
(44, 44, 'completed', 'RTX 4090', '2026-09-15 23:56:00', 263, 0.003, 16, 0.8494, 0.8462, 'fast_test'),
(45, 45, 'completed', 'A100', '2026-09-15 15:27:00', 212, 0.0003, 128, 0.9022, 0.8988, 'production_candidate'),
(46, 1, 'completed', 'RTX 4090', '2026-08-29 17:16:00', 59, 0.0001, 64, 0.8922, 0.8334, 'baseline'),
(47, 2, 'completed', 'A100', '2026-09-02 16:34:00', 10, 0.001, 128, 0.7452, 0.7328, 'class_weight'),
(48, 3, 'completed', 'CPU', '2026-09-12 05:26:00', 103, 5e-05, 64, 0.7942, 0.7628, 'debug'),
(49, 4, 'completed', 'A100', '2026-09-04 03:34:00', 72, 0.001, 128, 0.6711, 0.6258, 'debug'),
(50, 5, 'completed', 'H100', '2026-08-31 07:04:00', 50, 0.003, 16, 0.8237, 0.7683, 'debug');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(51, 6, 'completed', 'T4', '2026-09-04 16:37:00', 228, 0.0003, 64, 0.9587, 0.9498, 'class_weight'),
(52, 7, 'completed', 'A100', '2026-08-30 03:36:00', 257, 0.001, 32, 0.8795, 0.8905, 'production_candidate'),
(53, 8, 'running', 'L40S', '2026-08-30 11:57:00', 187, 0.003, 32, NULL, NULL, 'debug'),
(54, 9, 'failed', 'CPU', '2026-08-29 17:22:00', 143, 0.0001, 128, NULL, NULL, 'tuned'),
(55, 10, 'completed', 'L40S', '2026-09-18 17:49:00', 85, 0.001, 64, 0.7908, 0.7816, 'tuned'),
(56, 11, 'cancelled', 'CPU', '2026-08-27 08:58:00', 185, 0.01, 16, 0.665, NULL, 'tuned'),
(57, 12, 'completed', 'CPU', '2026-08-23 05:08:00', 9, 0.0001, 128, 0.8668, 0.861, 'production_candidate'),
(58, 13, 'completed', 'RTX 4090', '2026-09-18 11:48:00', 73, 0.003, 64, 0.9412, 0.951, 'tuned'),
(59, 14, 'completed', 'T4', '2026-09-19 16:06:00', 199, 0.0001, 32, 0.985, 0.9695, 'fast_test'),
(60, 15, 'completed', 'H100', '2026-08-28 09:45:00', 66, 0.01, 16, 0.8903, 0.8975, 'production_candidate'),
(61, 16, 'completed', 'L40S', '2026-08-29 18:01:00', 174, 0.0001, 16, 0.9750, 0.9697, 'class_weight'),
(62, 17, 'running', 'CPU', '2026-09-05 14:23:00', 127, 0.01, 16, NULL, NULL, 'class_weight'),
(63, 18, 'completed', 'T4', '2026-08-24 06:12:00', 61, 0.01, 128, 0.6972, 0.7132, 'class_weight'),
(64, 19, 'failed', 'H100', '2026-09-14 05:17:00', 94, 0.0003, 32, NULL, NULL, 'production_candidate'),
(65, 20, 'completed', 'A100', '2026-09-27 07:01:00', 111, 0.01, 64, 0.9291, 0.8695, 'production_candidate'),
(66, 21, 'completed', 'T4', '2026-09-16 10:11:00', 249, 0.001, 64, 0.9254, 0.9113, 'production_candidate'),
(67, 22, 'running', 'L40S', '2026-09-20 19:36:00', 133, 0.001, 32, NULL, NULL, 'fast_test'),
(68, 23, 'completed', 'RTX 4090', '2026-09-05 05:09:00', 163, 0.0003, 128, 0.841, 0.8127, 'production_candidate'),
(69, 24, 'completed', 'RTX 4090', '2026-08-30 10:44:00', 37, 0.0003, 128, 0.8406, 0.8204, 'tuned'),
(70, 25, 'completed', 'L40S', '2026-09-21 17:48:00', 127, 0.001, 16, 0.9452, 0.8863, 'class_weight'),
(71, 26, 'completed', 'A100', '2026-09-19 18:46:00', 278, 5e-05, 32, 0.9046, 0.9282, 'augmentation'),
(72, 27, 'completed', 'A100', '2026-08-28 12:42:00', 216, 0.0003, 32, 0.93, 0.8911, 'baseline'),
(73, 28, 'failed', 'A100', '2026-09-11 08:11:00', 92, 0.003, 16, NULL, NULL, 'production_candidate'),
(74, 29, 'failed', 'A100', '2026-09-18 11:17:00', 200, 0.003, 16, NULL, NULL, 'augmentation'),
(75, 30, 'completed', 'L40S', '2026-09-25 01:30:00', 252, 0.01, 32, 0.7148, 0.7305, 'production_candidate');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(76, 31, 'completed', 'A100', '2026-09-25 05:13:00', 58, 0.01, 128, 0.9061, 0.8563, 'class_weight'),
(77, 32, 'completed', 'L40S', '2026-09-07 05:45:00', 184, 0.0003, 128, 0.8698, 0.8467, 'debug'),
(78, 33, 'completed', 'H100', '2026-09-15 15:20:00', 231, 0.003, 32, 0.748, 0.712, 'debug'),
(79, 34, 'completed', 'L40S', '2026-09-03 08:28:00', 112, 0.0003, 16, 0.9388, 0.8891, 'augmentation'),
(80, 35, 'completed', 'H100', '2026-09-09 21:42:00', 137, 0.01, 128, 0.8894, 0.8389, 'production_candidate'),
(81, 36, 'completed', 'L40S', '2026-09-23 15:58:00', 148, 0.0001, 32, 0.7658, 0.7258, 'baseline'),
(82, 37, 'completed', 'RTX 4090', '2026-08-20 14:27:00', 114, 0.0001, 64, 0.8521, 0.8274, 'full_train'),
(83, 38, 'completed', 'T4', '2026-09-24 11:17:00', 279, 0.01, 16, 0.8205, 0.7627, 'production_candidate'),
(84, 39, 'completed', 'L40S', '2026-09-20 21:58:00', 234, 0.01, 64, 0.8972, 0.9012, 'baseline'),
(85, 40, 'completed', 'A100', '2026-09-01 18:09:00', 122, 0.003, 128, 0.7967, 0.7627, 'debug'),
(86, 41, 'running', 'CPU', '2026-09-13 04:34:00', 77, 0.0003, 16, NULL, 0.653, 'baseline'),
(87, 42, 'completed', 'L40S', '2026-09-12 07:50:00', 40, 0.001, 128, 0.8795, 0.8436, 'class_weight'),
(88, 43, 'cancelled', 'L40S', '2026-08-23 20:56:00', 252, 5e-05, 16, NULL, NULL, 'production_candidate'),
(89, 44, 'completed', 'T4', '2026-09-16 03:49:00', 30, 0.003, 64, 0.8064, 0.8085, 'baseline'),
(90, 45, 'cancelled', 'A100', '2026-09-18 10:12:00', 166, 0.0003, 16, NULL, NULL, 'production_candidate'),
(91, 1, 'failed', 'T4', '2026-08-29 21:26:00', 64, 0.003, 128, 0.661, NULL, 'debug'),
(92, 2, 'running', 'T4', '2026-09-03 18:07:00', 14, 0.0003, 32, 0.7359, NULL, 'full_train'),
(93, 3, 'completed', 'T4', '2026-09-10 20:48:00', 213, 0.001, 16, 0.9024, 0.8694, 'full_train'),
(94, 4, 'cancelled', 'CPU', '2026-09-04 04:37:00', 174, 5e-05, 64, NULL, NULL, 'baseline'),
(95, 5, 'completed', 'A100', '2026-09-01 10:21:00', 224, 0.0003, 16, 0.8767, 0.8807, 'full_train'),
(96, 6, 'completed', 'T4', '2026-09-07 14:48:00', 119, 0.01, 128, 0.8801, 0.8857, 'baseline'),
(97, 7, 'completed', 'H100', '2026-08-28 23:44:00', 96, 0.01, 32, 0.8206, 0.8271, 'augmentation'),
(98, 8, 'completed', 'L40S', '2026-09-01 09:07:00', 201, 0.0003, 16, 0.8307, 0.8086, 'debug'),
(99, 9, 'completed', 'A100', '2026-08-31 05:50:00', 44, 0.0001, 128, 0.8259, 0.7769, 'full_train'),
(100, 10, 'completed', 'L40S', '2026-09-17 18:05:00', 38, 0.003, 32, 0.8901, 0.8604, 'baseline');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(101, 11, 'cancelled', 'L40S', '2026-08-27 10:23:00', 118, 5e-05, 32, NULL, NULL, 'production_candidate'),
(102, 12, 'completed', 'A100', '2026-08-22 14:04:00', 42, 0.003, 128, 0.9582, 0.9481, 'class_weight'),
(103, 13, 'running', 'A100', '2026-09-19 13:27:00', 174, 0.001, 64, NULL, NULL, 'augmentation'),
(104, 14, 'completed', 'A100', '2026-09-17 12:27:00', 198, 0.0001, 16, 0.8879, 0.8348, 'tuned'),
(105, 15, 'completed', 'L40S', '2026-08-27 21:43:00', 180, 0.0003, 64, 0.9503, 0.9256, 'augmentation'),
(106, 16, 'completed', 'CPU', '2026-08-27 11:34:00', 264, 0.0001, 16, 0.8314, 0.8099, 'full_train'),
(107, 17, 'running', 'RTX 4090', '2026-09-05 09:59:00', 246, 0.001, 128, NULL, 0.5307, 'baseline'),
(108, 18, 'running', 'T4', '2026-08-25 09:53:00', 256, 0.003, 64, 0.7032, 0.7765, 'production_candidate'),
(109, 19, 'completed', 'H100', '2026-09-12 02:39:00', 125, 0.0003, 128, 0.7893, 0.7389, 'debug'),
(110, 20, 'failed', 'RTX 4090', '2026-09-25 19:44:00', 200, 0.0001, 128, NULL, NULL, 'class_weight'),
(111, 21, 'completed', 'T4', '2026-09-15 12:06:00', 40, 0.001, 64, 0.8211, 0.7703, 'augmentation'),
(112, 22, 'running', 'L40S', '2026-09-20 15:51:00', 194, 0.01, 128, NULL, NULL, 'production_candidate'),
(113, 23, 'cancelled', 'T4', '2026-09-08 06:43:00', 134, 0.01, 128, NULL, NULL, 'class_weight'),
(114, 24, 'completed', 'T4', '2026-08-27 10:11:00', 167, 5e-05, 16, 0.929, 0.8854, 'debug'),
(115, 25, 'cancelled', 'H100', '2026-09-25 06:37:00', 88, 0.001, 64, NULL, NULL, 'tuned'),
(116, 26, 'cancelled', 'A100', '2026-09-20 11:47:00', 181, 0.001, 16, NULL, NULL, 'debug'),
(117, 27, 'running', 'H100', '2026-08-30 03:22:00', 77, 0.0001, 128, 0.6632, NULL, 'production_candidate'),
(118, 28, 'running', 'RTX 4090', '2026-09-10 09:44:00', 63, 0.003, 32, 0.8057, 0.7689, 'fast_test'),
(119, 29, 'completed', 'H100', '2026-09-17 13:32:00', 182, 0.01, 128, 0.924, 0.9325, 'fast_test'),
(120, 30, 'completed', 'A100', '2026-09-23 14:33:00', 258, 0.003, 16, 0.8151, 0.7922, 'production_candidate'),
(121, 31, 'failed', 'T4', '2026-09-26 20:42:00', 175, 0.0003, 128, NULL, NULL, 'full_train'),
(122, 32, 'completed', 'CPU', '2026-09-06 02:59:00', 257, 0.0003, 128, 0.8297, 0.8276, 'class_weight'),
(123, 33, 'completed', 'T4', '2026-09-14 15:50:00', 104, 0.01, 64, 0.836, 0.8346, 'full_train'),
(124, 34, 'failed', 'H100', '2026-09-06 01:27:00', 151, 0.001, 16, NULL, NULL, 'full_train'),
(125, 35, 'cancelled', 'L40S', '2026-09-10 16:49:00', 28, 0.003, 128, NULL, NULL, 'production_candidate');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(126, 36, 'completed', 'T4', '2026-09-25 02:51:00', 162, 0.001, 128, 0.9637, 0.9614, 'debug'),
(127, 37, 'completed', 'A100', '2026-08-21 05:51:00', 194, 0.0003, 128, 0.8636, 0.813, 'augmentation'),
(128, 38, 'completed', 'RTX 4090', '2026-09-23 18:52:00', 231, 0.0001, 16, 0.8127, 0.8308, 'class_weight'),
(129, 39, 'completed', 'H100', '2026-09-23 20:51:00', 124, 0.01, 32, 0.9518, 0.9502, 'tuned'),
(130, 40, 'cancelled', 'A100', '2026-09-02 11:11:00', 71, 0.001, 32, NULL, 0.7857, 'class_weight'),
(131, 41, 'completed', 'RTX 4090', '2026-09-14 04:34:00', 189, 0.003, 64, 0.8516, 0.8672, 'full_train'),
(132, 42, 'running', 'A100', '2026-09-13 13:13:00', 243, 0.0001, 64, NULL, NULL, 'fast_test'),
(133, 43, 'failed', 'H100', '2026-08-25 06:58:00', 9, 5e-05, 64, 0.6604, NULL, 'debug'),
(134, 44, 'completed', 'H100', '2026-09-15 21:08:00', 148, 0.01, 32, 0.8723, 0.8665, 'tuned'),
(135, 45, 'running', 'RTX 4090', '2026-09-17 09:17:00', 31, 5e-05, 64, NULL, NULL, 'tuned'),
(136, 1, 'completed', 'A100', '2026-08-31 04:10:00', 178, 0.01, 16, 0.8106, 0.7812, 'augmentation'),
(137, 2, 'completed', 'A100', '2026-09-01 10:27:00', 78, 0.001, 64, 0.8903, 0.8368, 'baseline'),
(138, 3, 'failed', 'L40S', '2026-09-14 01:20:00', 214, 0.003, 32, NULL, NULL, 'tuned'),
(139, 4, 'completed', 'A100', '2026-09-03 01:42:00', 134, 0.01, 16, 0.7451, 0.7006, 'debug'),
(140, 5, 'failed', 'A100', '2026-08-31 22:56:00', 50, 5e-05, 32, NULL, NULL, 'baseline'),
(141, 6, 'completed', 'RTX 4090', '2026-09-05 05:04:00', 61, 5e-05, 64, 0.8535, 0.8105, 'class_weight'),
(142, 7, 'completed', 'L40S', '2026-08-29 02:37:00', 271, 0.0001, 32, 0.9415, 0.8897, 'debug'),
(143, 8, 'failed', 'H100', '2026-08-29 13:20:00', 73, 0.0003, 16, 0.6308, NULL, 'tuned'),
(144, 9, 'completed', 'H100', '2026-09-01 05:32:00', 77, 0.0003, 64, 0.9109, 0.877, 'baseline'),
(145, 10, 'failed', 'L40S', '2026-09-19 15:42:00', 53, 0.003, 32, NULL, 0.6021, 'full_train'),
(146, 11, 'completed', 'L40S', '2026-08-28 22:11:00', 238, 0.0001, 16, 0.959, 0.9475, 'debug'),
(147, 12, 'completed', 'H100', '2026-08-26 11:57:00', 101, 0.0001, 64, 0.9588, 0.9428, 'full_train'),
(148, 13, 'running', 'T4', '2026-09-18 00:24:00', 244, 0.01, 64, 0.8191, NULL, 'full_train'),
(149, 14, 'completed', 'H100', '2026-09-16 12:07:00', 154, 0.001, 128, 0.8619, 0.8500, 'tuned'),
(150, 15, 'completed', 'RTX 4090', '2026-08-28 15:05:00', 106, 0.0001, 128, 0.8319, 0.8462, 'tuned');

INSERT INTO runs (id, experiment_id, status, gpu, started_at, duration_min, learning_rate, batch_size, accuracy, f1_score, notes) VALUES
(151, 16, 'completed', 'RTX 4090', '2026-08-27 10:24:00', 23, 5e-05, 16, 0.9255, 0.866, 'full_train'),
(152, 17, 'completed', 'H100', '2026-09-04 14:03:00', 213, 0.01, 64, 0.8727, 0.8727, 'production_candidate'),
(153, 18, 'failed', 'RTX 4090', '2026-08-23 10:24:00', 51, 0.0001, 16, NULL, NULL, 'augmentation'),
(154, 19, 'completed', 'T4', '2026-09-14 19:46:00', 268, 0.0001, 16, 0.7997, 0.7474, 'class_weight'),
(155, 20, 'failed', 'H100', '2026-09-24 09:37:00', 28, 0.0001, 64, NULL, NULL, 'debug'),
(156, 21, 'failed', 'A100', '2026-09-14 13:14:00', 234, 0.0001, 128, NULL, NULL, 'tuned'),
(157, 22, 'running', 'A100', '2026-09-18 12:31:00', 133, 0.0003, 64, NULL, NULL, 'production_candidate'),
(158, 23, 'completed', 'A100', '2026-09-06 01:54:00', 52, 5e-05, 128, 0.9058, 0.8876, 'debug'),
(159, 24, 'running', 'A100', '2026-08-27 12:57:00', 248, 0.001, 32, NULL, NULL, 'baseline'),
(160, 45, 'completed', 'H100', '2026-09-25 12:30:00', 74, 0.0001, 32, 0.9721, 0.9654, 'production_candidate');

CREATE VIEW run_overview AS
SELECT
    r.id AS run_id,
    e.id AS experiment_id,
    e.title AS experiment_title,
    rs.full_name AS researcher_name,
    rs.team AS researcher_team,
    m.name AS model_name,
    m.family AS model_family,
    m.task_type AS task_type,
    d.name AS dataset_name,
    d.source_type AS dataset_source,
    d.records_count AS dataset_records,
    r.status,
    r.gpu,
    r.started_at,
    r.duration_min,
    r.learning_rate,
    r.batch_size,
    r.accuracy,
    r.f1_score,
    r.notes
FROM runs r
JOIN experiments e ON e.id = r.experiment_id
JOIN researchers rs ON rs.id = e.researcher_id
JOIN models m ON m.id = e.model_id
JOIN datasets d ON d.id = e.dataset_id;

-- Проверка установки: должно вернуть 160
SELECT COUNT(*) AS rows_in_lab FROM run_overview;