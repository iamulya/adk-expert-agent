export function deepClone<T>(value: T): T {
  if (typeof value !== "object" || value === null) {
    return value;
  }
  if (value instanceof Date) {
    return new Date(value.getTime()) as any;
  }
  if (value instanceof Array) {
    return value.reduce((arr, item, i) => {
      arr[i] = deepClone(item);
      return arr;
    }, []) as any;
  }
  if (value instanceof Object) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return Object.keys(value).reduce((obj: any, key) => {
      obj[key] = deepClone((value as Record<string, unknown>)[key]);
      return obj;
    }, {} as T);
  }
  return value;
}
