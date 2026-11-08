-- =====================================================================
-- Healthcare Access & Resource Optimization Tracker  |  SQL analysis
-- Database: hospital_ops.db (SQLite)
-- Tables: admissions, wards, staff_shifts, daily_occupancy
-- NOTE: timestamps, beds and staffing are SIMULATED (see README).
-- =====================================================================


-- ---------------------------------------------------------------------
-- Q1. Occupancy summary by ward: average, peak, and number of "red" days
--     (red = occupancy above 90%)
-- ---------------------------------------------------------------------
SELECT
    ward_id,
    MAX(total_beds)                                   AS total_beds,
    ROUND(AVG(occupancy_pct), 1)                      AS avg_occupancy_pct,
    ROUND(MAX(occupancy_pct), 1)                      AS peak_occupancy_pct,
    SUM(CASE WHEN occupancy_pct > 90 THEN 1 ELSE 0 END) AS red_days,
    COUNT(*)                                          AS days_observed
FROM daily_occupancy
GROUP BY ward_id
ORDER BY red_days DESC
LIMIT 15;


-- ---------------------------------------------------------------------
-- Q2. Current ward status with a traffic-light alert (latest day in data)
--     green < 75%, yellow 75-90%, red > 90%
-- ---------------------------------------------------------------------
SELECT
    ward_id,
    occupied_beds,
    total_beds,
    ROUND(occupancy_pct, 1) AS occupancy_pct,
    CASE
        WHEN occupancy_pct > 90 THEN 'RED'
        WHEN occupancy_pct >= 75 THEN 'YELLOW'
        ELSE 'GREEN'
    END AS alert
FROM daily_occupancy
WHERE date = (SELECT MAX(date) FROM daily_occupancy)
ORDER BY occupancy_pct DESC
LIMIT 15;


-- ---------------------------------------------------------------------
-- Q3. Peak admission hours: admissions per hour of day with share of total
--     and rank (window functions)
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT admit_hour, COUNT(*) AS admissions
    FROM admissions
    GROUP BY admit_hour
)
SELECT
    admit_hour,
    admissions,
    ROUND(100.0 * admissions / SUM(admissions) OVER (), 1) AS pct_of_total,
    RANK() OVER (ORDER BY admissions DESC)                  AS peak_rank
FROM hourly
ORDER BY admit_hour;


-- ---------------------------------------------------------------------
-- Q4. Admissions by day of week and admission type
--     (admit_dow: 0 = Monday ... 6 = Sunday)
-- ---------------------------------------------------------------------
SELECT
    CASE admit_dow
        WHEN 0 THEN 'Mon' WHEN 1 THEN 'Tue' WHEN 2 THEN 'Wed'
        WHEN 3 THEN 'Thu' WHEN 4 THEN 'Fri' WHEN 5 THEN 'Sat' ELSE 'Sun'
    END AS weekday,
    SUM(CASE WHEN type_of_admission = 'Emergency' THEN 1 ELSE 0 END) AS emergency,
    SUM(CASE WHEN type_of_admission = 'Trauma'    THEN 1 ELSE 0 END) AS trauma,
    SUM(CASE WHEN type_of_admission = 'Urgent'    THEN 1 ELSE 0 END) AS urgent,
    COUNT(*) AS total
FROM admissions
GROUP BY admit_dow
ORDER BY admit_dow;


-- ---------------------------------------------------------------------
-- Q5. Patients per doctor and per nurse, by ward and shift
--     (average occupied beds over the last 30 days / staff on that shift)
--     Higher ratio = heavier workload.
-- ---------------------------------------------------------------------
WITH recent AS (
    SELECT ward_id, AVG(occupied_beds) AS avg_patients
    FROM daily_occupancy
    WHERE date > DATE((SELECT MAX(date) FROM daily_occupancy), '-30 days')
    GROUP BY ward_id
)
SELECT
    s.ward_id,
    s.shift,
    ROUND(r.avg_patients, 0)               AS avg_patients,
    s.doctors,
    s.nurses,
    ROUND(r.avg_patients / s.doctors, 1)   AS patients_per_doctor,
    ROUND(r.avg_patients / s.nurses, 1)    AS patients_per_nurse
FROM staff_shifts s
JOIN recent r ON r.ward_id = s.ward_id
ORDER BY patients_per_doctor DESC
LIMIT 15;


-- ---------------------------------------------------------------------
-- Q6. Average length of stay by age band and illness severity
--     (uses bin-midpoint LOS in days)
-- ---------------------------------------------------------------------
SELECT
    age,
    ROUND(AVG(CASE WHEN severity_of_illness = 'Minor'    THEN los_days_mid END), 1) AS minor,
    ROUND(AVG(CASE WHEN severity_of_illness = 'Moderate' THEN los_days_mid END), 1) AS moderate,
    ROUND(AVG(CASE WHEN severity_of_illness = 'Extreme'  THEN los_days_mid END), 1) AS extreme,
    ROUND(AVG(los_days_mid), 1)                                                     AS overall,
    COUNT(*)                                                                         AS patients
FROM admissions
GROUP BY age
ORDER BY CAST(SUBSTR(age, 1, INSTR(age, '-') - 1) AS INTEGER);


-- ---------------------------------------------------------------------
-- Q7. Average length of stay by department and admission type, with the
--     share of long stays (over 30 days)
-- ---------------------------------------------------------------------
SELECT
    department,
    type_of_admission,
    COUNT(*)                                                       AS patients,
    ROUND(AVG(los_days_mid), 1)                                    AS avg_los_days,
    ROUND(100.0 * SUM(CASE WHEN los_days_mid > 30 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_long_stay
FROM admissions
GROUP BY department, type_of_admission
ORDER BY avg_los_days DESC;


-- ---------------------------------------------------------------------
-- Q8. Rolling 7-day average occupancy per hospital
--     Step 1: hospital-level daily occupancy. Step 2: 7-day window.
-- ---------------------------------------------------------------------
WITH hospital_daily AS (
    SELECT
        w.hospital_code,
        d.date,
        100.0 * SUM(d.occupied_beds) / SUM(d.total_beds) AS occupancy_pct
    FROM daily_occupancy d
    JOIN wards w ON w.ward_id = d.ward_id
    GROUP BY w.hospital_code, d.date
)
SELECT
    hospital_code,
    DATE(date) AS day,
    ROUND(occupancy_pct, 1) AS daily_pct,
    ROUND(AVG(occupancy_pct) OVER (
        PARTITION BY hospital_code
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_7d_pct
FROM hospital_daily
WHERE hospital_code = 2
ORDER BY date DESC
LIMIT 14;


-- ---------------------------------------------------------------------
-- Q9. Month-over-month change in admissions per hospital
--     (LAG window function)
-- ---------------------------------------------------------------------
WITH monthly AS (
    SELECT
        hospital_code,
        STRFTIME('%Y-%m', admit_ts) AS month,
        COUNT(*) AS admissions
    FROM admissions
    GROUP BY hospital_code, month
)
SELECT
    hospital_code,
    month,
    admissions,
    LAG(admissions) OVER (PARTITION BY hospital_code ORDER BY month) AS prev_month,
    ROUND(100.0 * (admissions - LAG(admissions) OVER (PARTITION BY hospital_code ORDER BY month))
          / LAG(admissions) OVER (PARTITION BY hospital_code ORDER BY month), 1) AS pct_change
FROM monthly
WHERE hospital_code = 2
ORDER BY month;


-- ---------------------------------------------------------------------
-- Q10. Risk ranking: wards that are BOTH often over capacity AND heavily
--      loaded per doctor. Combines occupancy and staffing into one score.
-- ---------------------------------------------------------------------
WITH occ AS (
    SELECT
        ward_id,
        AVG(occupancy_pct) AS avg_occ,
        100.0 * SUM(CASE WHEN occupancy_pct > 90 THEN 1 ELSE 0 END) / COUNT(*) AS pct_red_days,
        AVG(occupied_beds) AS avg_patients
    FROM daily_occupancy
    GROUP BY ward_id
),
doc AS (
    SELECT ward_id, MIN(doctors) AS min_doctors   -- thinnest shift
    FROM staff_shifts
    GROUP BY ward_id
)
SELECT
    o.ward_id,
    ROUND(o.avg_occ, 1)                          AS avg_occ_pct,
    ROUND(o.pct_red_days, 1)                     AS pct_red_days,
    ROUND(o.avg_patients / d.min_doctors, 1)     AS patients_per_doctor_worst_shift,
    RANK() OVER (ORDER BY o.pct_red_days DESC)   AS occupancy_rank,
    RANK() OVER (ORDER BY o.avg_patients / d.min_doctors DESC) AS staffing_rank
FROM occ o
JOIN doc d ON d.ward_id = o.ward_id
ORDER BY (RANK() OVER (ORDER BY o.pct_red_days DESC)
        + RANK() OVER (ORDER BY o.avg_patients / d.min_doctors DESC))
LIMIT 10;
