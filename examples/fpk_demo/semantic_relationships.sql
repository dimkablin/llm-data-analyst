BEGIN;

CREATE TABLE IF NOT EXISTS demo_fpk.branches (
    branch_id smallint PRIMARY KEY,
    branch_name text NOT NULL UNIQUE,
    stats_code text NOT NULL UNIQUE,
    csi_code text NOT NULL UNIQUE,
    is_total boolean NOT NULL DEFAULT false
);

INSERT INTO demo_fpk.branches (branch_id, branch_name, stats_code, csi_code, is_total)
VALUES
    (0, 'АО «ФПК» (итог)', 'ФПК', 'ФПК', true),
    (1, 'Северо-Западный филиал', 'СЕВ.ЗАП.', 'С-ЗАП', false),
    (2, 'Московский филиал', 'МОСК.', 'МОСК', false),
    (3, 'Горьковский филиал', 'ГОРЬК.', 'ГОРЬК', false),
    (4, 'Северо-Кавказский филиал', 'С-КАВ.', 'С-КАВ', false),
    (5, 'Приволжский филиал', 'ПРИВ.', 'ПРИВ', false),
    (6, 'Куйбышевский филиал', 'КУЙБ.', 'КБШ', false),
    (7, 'Уральский филиал', 'СВЕРД.', 'УР', false),
    (8, 'Западно-Сибирский филиал', 'З-СИБ.', 'З-СИБ', false),
    (9, 'Восточно-Сибирский филиал', 'В-СИБ.', 'В-СИБ', false),
    (10, 'Дальневосточный филиал', 'ДВОСТ.', 'ДВОСТ', false)
ON CONFLICT (branch_id) DO UPDATE SET
    branch_name = EXCLUDED.branch_name,
    stats_code = EXCLUDED.stats_code,
    csi_code = EXCLUDED.csi_code,
    is_total = EXCLUDED.is_total;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM demo_fpk.stat_stats AS source
        LEFT JOIN demo_fpk.branches AS branch ON branch.stats_code = source.structure
        WHERE branch.branch_id IS NULL
    ) THEN
        RAISE EXCEPTION 'demo_fpk.branches does not cover every stat_stats.structure value';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM demo_fpk.stat_csi AS source
        LEFT JOIN demo_fpk.branches AS branch ON branch.csi_code = source.structure
        WHERE branch.branch_id IS NULL
    ) THEN
        RAISE EXCEPTION 'demo_fpk.branches does not cover every stat_csi.structure value';
    END IF;
END
$$;

CREATE OR REPLACE VIEW demo_fpk.stat_stats_semantic AS
SELECT source.*, branch.branch_id
FROM demo_fpk.stat_stats AS source
JOIN demo_fpk.branches AS branch ON branch.stats_code = source.structure
WHERE NOT branch.is_total;

CREATE OR REPLACE VIEW demo_fpk.stat_csi_semantic AS
SELECT DISTINCT ON (source.date, source.structure, source.type, source.sys_section)
    source.*,
    branch.branch_id
FROM demo_fpk.stat_csi AS source
JOIN demo_fpk.branches AS branch ON branch.csi_code = source.structure
WHERE NOT branch.is_total
  AND source.type = 'fact'
ORDER BY
    source.date,
    source.structure,
    source.type,
    source.sys_section,
    source.add_time DESC;

COMMENT ON TABLE demo_fpk.branches IS
    'Канонический справочник филиалов для сопоставления кодов stats и CSI.';
COMMENT ON VIEW demo_fpk.stat_stats_semantic IS
    'Операционная статистика с каноническим branch_id.';
COMMENT ON VIEW demo_fpk.stat_csi_semantic IS
    'Последняя версия CSI с каноническим branch_id.';

COMMIT;
