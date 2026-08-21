\set ON_ERROR_STOP on
\encoding UTF8

BEGIN;
TRUNCATE stg.stat_stats, stg.stat_csi, stg.stat_isoo, stg.stat_manual;

\copy stg.stat_stats (pass_turnover, pass_count, car_turnover, seat_turnover, structure, date, sys_section, metric, add_time, updated_at, is_deleted) FROM '/tmp/fpk_seed/stg_stat_stats.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
\copy stg.stat_csi (type, "безопасность", "дорожный_набор", "поездка_с_детьми", "покупка_билетов", "ирс_попутчик", "постельные_принадлежности", "предоплаченное_питание", "ржд_бонус", "работа_проводников", "санитарное_состояние", "стоимость_поездки", "техническое_состояние", "уровень_комфорта", "услуги_вагона_ресторана", date, "индекс_удовлетворенности_пас", add_time, structure, "индекс_потребительской_лояльност", sys_section) FROM '/tmp/fpk_seed/stg_stat_csi.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
\copy stg.stat_isoo ("шифр", "тематика_обращения", "с_зап", "моск", "горьк", "с_кав", "прив", "кбш", "ур", "з_сиб", "в_сиб", "двост", "всего_по_филиалам", "еисц", "почта_оао_ржд", "портал_генерального_директора_оао", "почта_ао_фпк", "почта_генерального_директора_ао_ф", "всего_по_каналам_поступления_обра", date, sys_section, add_time) FROM '/tmp/fpk_seed/stg_stat_isoo.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
\copy stg.stat_manual (value_name, value, date, sys_section, metric, add_time) FROM '/tmp/fpk_seed/stg_stat_manual.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')

DO $$
BEGIN
    IF (SELECT count(*) FROM stg.stat_stats) <> 8030
       OR (SELECT count(*) FROM stg.stat_csi) <> 1200
       OR (SELECT count(*) FROM stg.stat_isoo) <> 9056
       OR (SELECT count(*) FROM stg.stat_manual) <> 748 THEN
        RAISE EXCEPTION 'Unexpected FPK demo row count';
    END IF;
END
$$;

ANALYZE stg.stat_stats;
ANALYZE stg.stat_csi;
ANALYZE stg.stat_isoo;
ANALYZE stg.stat_manual;
COMMIT;
