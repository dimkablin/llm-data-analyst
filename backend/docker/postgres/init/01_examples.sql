-- Load example CSVs (paths are inside the container: /examples is mounted read-only).
-- Runs once on first volume init via official postgres entrypoint (psql -f).

CREATE SCHEMA IF NOT EXISTS examples;

-- Titanic (Kaggle-style)
CREATE TABLE examples.titanic (
    "PassengerId" INTEGER NOT NULL PRIMARY KEY,
    "Survived" INTEGER NOT NULL,
    "Pclass" INTEGER NOT NULL,
    "Name" TEXT NOT NULL,
    "Sex" TEXT NOT NULL,
    "Age" DOUBLE PRECISION,
    "SibSp" INTEGER NOT NULL,
    "Parch" INTEGER NOT NULL,
    "Ticket" TEXT NOT NULL,
    "Fare" DOUBLE PRECISION NOT NULL,
    "Cabin" TEXT,
    "Embarked" TEXT
);

\copy examples.titanic ("PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked") FROM '/examples/titanic/dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '')

-- Video game sales (Year may be N/A → TEXT)
CREATE TABLE examples.video_game_sales (
    sales_rank INTEGER NOT NULL,
    name TEXT NOT NULL,
    platform TEXT NOT NULL,
    year TEXT NOT NULL,
    genre TEXT NOT NULL,
    publisher TEXT NOT NULL,
    na_sales DOUBLE PRECISION NOT NULL,
    eu_sales DOUBLE PRECISION NOT NULL,
    jp_sales DOUBLE PRECISION NOT NULL,
    other_sales DOUBLE PRECISION NOT NULL,
    global_sales DOUBLE PRECISION NOT NULL
);

\copy examples.video_game_sales (sales_rank, name, platform, year, genre, publisher, na_sales, eu_sales, jp_sales, other_sales, global_sales) FROM '/examples/video-game-sales/dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '')

-- Bank churn: last two CSV columns exceed PostgreSQL 63-char identifier limit → short names
CREATE TABLE examples.bank_churn_clients (
    clientnum BIGINT NOT NULL PRIMARY KEY,
    attrition_flag TEXT NOT NULL,
    customer_age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    dependent_count INTEGER NOT NULL,
    education_level TEXT NOT NULL,
    marital_status TEXT NOT NULL,
    income_category TEXT NOT NULL,
    card_category TEXT NOT NULL,
    months_on_book INTEGER NOT NULL,
    total_relationship_count INTEGER NOT NULL,
    months_inactive_12_mon INTEGER NOT NULL,
    contacts_count_12_mon INTEGER NOT NULL,
    credit_limit DOUBLE PRECISION NOT NULL,
    total_revolving_bal DOUBLE PRECISION NOT NULL,
    avg_open_to_buy DOUBLE PRECISION NOT NULL,
    total_amt_chng_q4_q1 DOUBLE PRECISION NOT NULL,
    total_trans_amt DOUBLE PRECISION NOT NULL,
    total_trans_ct INTEGER NOT NULL,
    total_ct_chng_q4_q1 DOUBLE PRECISION NOT NULL,
    avg_utilization_ratio DOUBLE PRECISION NOT NULL,
    nb_classifier_attrition_contacts_dep_edu_inactive_1 DOUBLE PRECISION NOT NULL,
    nb_classifier_attrition_contacts_dep_edu_inactive_2 DOUBLE PRECISION NOT NULL
);

\copy examples.bank_churn_clients (clientnum, attrition_flag, customer_age, gender, dependent_count, education_level, marital_status, income_category, card_category, months_on_book, total_relationship_count, months_inactive_12_mon, contacts_count_12_mon, credit_limit, total_revolving_bal, avg_open_to_buy, total_amt_chng_q4_q1, total_trans_amt, total_trans_ct, total_ct_chng_q4_q1, avg_utilization_ratio, nb_classifier_attrition_contacts_dep_edu_inactive_1, nb_classifier_attrition_contacts_dep_edu_inactive_2) FROM '/examples/bank_churn_clients/dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8', NULL '')
