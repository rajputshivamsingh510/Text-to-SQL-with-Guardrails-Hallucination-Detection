-- Creates the least-privilege role the API should connect as in production.
-- Run ONCE as the database owner, AFTER the tables exist (re-run it if you re-seed, since seeding recreates tables):
--
--   psql "$ADMIN_DATABASE_URL" -v ro_password='choose-a-strong-password' -f scripts/create_readonly_role.sql
--
-- Then set the app's DATABASE_URL to the same host/database with user t2sql_readonly and that password.
-- Even if every application-level guardrail failed, this role physically cannot write, cannot read the
-- PII columns (customers.email / customers.phone), and cannot run a query for longer than 5 seconds.

SELECT 'CREATE ROLE t2sql_readonly LOGIN'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 't2sql_readonly')\gexec

ALTER ROLE t2sql_readonly WITH LOGIN PASSWORD :'ro_password';

DO $$ BEGIN
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO t2sql_readonly', current_database());
END $$;

GRANT USAGE ON SCHEMA public TO t2sql_readonly;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM t2sql_readonly;

GRANT SELECT ON products, orders, order_items TO t2sql_readonly;
-- Column-level grant: email and phone are deliberately NOT included.
GRANT SELECT (id, name, country, city, signup_date) ON customers TO t2sql_readonly;

ALTER ROLE t2sql_readonly SET statement_timeout = '5s';
ALTER ROLE t2sql_readonly SET default_transaction_read_only = on;
ALTER ROLE t2sql_readonly SET idle_in_transaction_session_timeout = '10s';
