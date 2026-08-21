import type { QueryExecutionOptions } from './backend-types';

type StorageLike = Pick<Storage, 'getItem' | 'setItem'>;

function storageKey(sessionId: string): string {
  return `slash_execution_${sessionId}`;
}

export function saveQueryExecutionOptions(
  storage: StorageLike,
  sessionId: string,
  query: string,
  options: QueryExecutionOptions,
): void {
  storage.setItem(storageKey(sessionId), JSON.stringify({ query, options }));
}

export function loadQueryExecutionOptions(
  storage: StorageLike,
  sessionId: string,
  query: string,
): QueryExecutionOptions {
  try {
    const parsed = JSON.parse(storage.getItem(storageKey(sessionId)) ?? 'null') as {
      query?: unknown;
      options?: QueryExecutionOptions;
    } | null;
    return parsed?.query === query && parsed.options ? parsed.options : {};
  } catch {
    return {};
  }
}
