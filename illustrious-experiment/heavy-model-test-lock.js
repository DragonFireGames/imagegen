const LOCK_KEY = "__illustriousHeavyModelTest";

export function acquireHeavyModelTest(name) {
  const active = globalThis[LOCK_KEY];
  if (active) {
    throw new Error("Another memory-heavy model test is already running: " + active +
      ". Let it finish before starting " + name + ".");
  }
  globalThis[LOCK_KEY] = name;
}

export function releaseHeavyModelTest(name) {
  if (globalThis[LOCK_KEY] === name) {
    delete globalThis[LOCK_KEY];
  }
}
