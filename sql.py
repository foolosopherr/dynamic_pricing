WITH
    arrayMap(x -> toDate(x), [
        '2025-01-06',
        '2025-01-13',
        '2025-01-20'
    ]) AS monday_list,
    [101, 205, 330] AS store_list,
    [('MON_THU', -1), ('FRI', 3), ('SAT_SUN', 4)] AS bucket_defs

SELECT
    b.product_code,
    b.monday_date,
    b.bucket_now,
    argMax(sf.sale_price, sf.trade_dt) AS sale_price_prev_bucket,
    argMax(sf.base_price, sf.trade_dt) AS base_price_prev_bucket
FROM
(
    SELECT
        p.product_code,
        m.monday_date,
        bd.1 AS bucket_now,
        addDays(m.monday_date, bd.2) AS prev_bucket_last_day
    FROM
        (SELECT DISTINCT product_code
         FROM sales
         WHERE store_id IN store_list
           AND segment_code IN (1, 3)
        ) AS p
    CROSS JOIN (SELECT arrayJoin(monday_list) AS monday_date) AS m
    CROSS JOIN (SELECT arrayJoin(bucket_defs) AS bd) AS d
) AS b
LEFT JOIN
(
    SELECT
        product_code,
        trade_dt,
        sale_price,
        base_price
    FROM sales
    WHERE store_id IN store_list
      AND segment_code IN (1, 3)
) AS sf
    ON sf.product_code = b.product_code
   AND sf.trade_dt <= b.prev_bucket_last_day
GROUP BY
    b.product_code,
    b.monday_date,
    b.bucket_now
ORDER BY
    b.monday_date,
    b.product_code,
    b.bucket_now
