import type {
  DBConnection,
  DBConnectionFormPayload,
  DBConnectionType,
} from "../../lib/backend-types";

export type SecretMode = "keep" | "replace" | "clear";

export type ConnectionFormState = {
  name: string;
  dbType: DBConnectionType;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  secretMode: SecretMode;
  sslmode: string;
  secure: boolean;
};

export const DEFAULT_FORM: ConnectionFormState = {
  name: "",
  dbType: "postgresql",
  host: "",
  port: "5432",
  database: "",
  username: "",
  password: "",
  secretMode: "replace",
  sslmode: "prefer",
  secure: false,
};

export function defaultPortFor(dbType: DBConnectionType): string {
  return dbType === "clickhouse" ? "8123" : "5432";
}

export function toFormState(connection?: DBConnection | null): ConnectionFormState {
  if (!connection) {
    return DEFAULT_FORM;
  }
  return {
    name: connection.name,
    dbType: connection.db_type,
    host: connection.host,
    port: connection.port ? String(connection.port) : defaultPortFor(connection.db_type),
    database: connection.database ?? "",
    username: connection.username ?? "",
    password: "",
    secretMode: "keep",
    sslmode:
      connection.db_type === "postgresql"
        ? String(connection.options_json?.sslmode ?? "prefer")
        : "prefer",
    secure: Boolean(connection.options_json?.secure),
  };
}

export function buildPayload(
  form: ConnectionFormState,
  editing: boolean,
  existingConnection?: DBConnection | null,
): DBConnectionFormPayload {
  const existingSchema = typeof existingConnection?.options_json?.schema === "string"
    ? existingConnection.options_json.schema
    : null;
  const payload: DBConnectionFormPayload = {
    name: form.name.trim(),
    db_type: form.dbType,
    host: form.host.trim(),
    port: form.port.trim() ? Number(form.port.trim()) : null,
    database: form.database.trim() || null,
    username: form.username.trim() || null,
    options_json:
      form.dbType === "postgresql"
        ? { sslmode: form.sslmode, schema: existingSchema }
        : { secure: form.secure, schema: existingSchema },
  };

  if (!editing || form.secretMode === "replace") {
    payload.password = form.password || null;
  }
  if (editing && form.secretMode === "clear") {
    payload.clear_password = true;
  }
  return payload;
}
