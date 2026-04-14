# Chart Overflow Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Предотвратить выход графиков Plotly за границы карточек-контейнеров на странице «Визуализации».

**Architecture:** Три уровня CSS-контейнеров позволяют контенту переполняться: grid-item (`motion.div`) не имеет `min-w-0`, карточка (`article`) не имеет `overflow-hidden`, а сам div графика не ограничивает overflow. Достаточно добавить правильные Tailwind-классы в двух файлах.

**Tech Stack:** React, Tailwind CSS, Plotly.js (`plotly.js-dist-min`)

---

## Диагностика (root cause)

```
DashboardPanel.tsx — grid-item motion.div
  └─ ArtifactSurface.tsx — <article> card wrapper      ← нет overflow-hidden
       └─ PlotArtifact — <div ref={containerRef}>       ← нет overflow-hidden / min-w-0
            └─ Plotly SVG / canvas                      ← может расти шире родителя
```

### Причины переполнения

| # | Место | Проблема |
|---|-------|----------|
| 1 | `DashboardPanel.tsx:509` `<motion.div>` | Grid-child без `min-w-0` — в CSS Grid дочерние элементы по умолчанию `min-width: auto`, что позволяет им расширять колонку за её пределы |
| 2 | `ArtifactSurface.tsx:393` `<article>` | Нет `overflow-hidden` — внутренние абсолютно позиционированные элементы Plotly (tooltip, legend) вылезают за скруглённые углы |
| 3 | `ArtifactSurface.tsx:379` `<div ref={containerRef}>` | Нет `overflow-hidden` — Plotly SVG иногда рендерится шире контейнера при первой отрисовке до `resize` |

---

## File Map

| Файл | Изменение |
|------|-----------|
| `frontend/src/app/components/workspace/ArtifactSurface.tsx` | Строки 379, 393 — добавить `overflow-hidden min-w-0` |
| `frontend/src/app/components/workspace/DashboardPanel.tsx` | Строка 509 — добавить `min-w-0` на `motion.div` |

---

## Task 1: Ограничить переполнение на уровне карточки и графика

**Files:**
- Modify: `frontend/src/app/components/workspace/ArtifactSurface.tsx:379`
- Modify: `frontend/src/app/components/workspace/ArtifactSurface.tsx:393`

- [ ] **Step 1: Открыть файл и найти строки**

Убедиться, что строки соответствуют:

```
379: className="h-[400px] w-full rounded-2xl border border-border/40"
393: className="rounded-3xl border border-border/50 bg-card p-5 shadow-sm"
```

- [ ] **Step 2: Добавить `overflow-hidden` на контейнер графика (строка 379)**

Было:
```tsx
<div
  ref={containerRef}
  className="h-[400px] w-full rounded-2xl border border-border/40"
/>
```

Стало:
```tsx
<div
  ref={containerRef}
  className="h-[400px] w-full min-w-0 overflow-hidden rounded-2xl border border-border/40"
/>
```

- [ ] **Step 3: Добавить `overflow-hidden` на карточку-обёртку (строка 393)**

Было:
```tsx
<article className="rounded-3xl border border-border/50 bg-card p-5 shadow-sm">
```

Стало:
```tsx
<article className="overflow-hidden rounded-3xl border border-border/50 bg-card p-5 shadow-sm">
```

- [ ] **Step 4: Сохранить файл и убедиться что другие места в файле не затронуты**

---

## Task 2: Исправить grid-item, чтобы колонка не расширялась

**Files:**
- Modify: `frontend/src/app/components/workspace/DashboardPanel.tsx:509`

- [ ] **Step 1: Найти строку с `motion.div` внутри grid**

```tsx
// ~строка 509
<motion.div key={artifact.id} layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
```

- [ ] **Step 2: Добавить `min-w-0` на `motion.div`**

Было:
```tsx
<motion.div key={artifact.id} layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
```

Стало:
```tsx
<motion.div key={artifact.id} layout className="min-w-0" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
```

- [ ] **Step 3: Сохранить и убедиться что пропсы `motion.div` сохранены полностью**

---

## Task 3: Визуальная проверка в браузере

- [ ] **Step 1: Запустить дев-сервер если не запущен**

```bash
cd frontend && npm run dev
```

- [ ] **Step 2: Открыть страницу «Визуализации» (вкладка с графиками)**

Проверить что:
- [ ] Карточки не выходят за пределы grid-колонок
- [ ] Скруглённые углы `rounded-3xl` обрезают содержимое графика
- [ ] При изменении ширины окна (responsive) графики масштабируются внутри карточек

- [ ] **Step 3: Проверить тёмную и светлую темы**

Убедиться что `overflow-hidden` не обрезает legend/tooltip в нормальном состоянии.

---

## Task 4: Commit

- [ ] **Step 1: Staged changes**

```bash
git add frontend/src/app/components/workspace/ArtifactSurface.tsx
git add frontend/src/app/components/workspace/DashboardPanel.tsx
```

- [ ] **Step 2: Commit**

```bash
git commit -m "fix(ui): prevent Plotly charts from overflowing card boundaries

Add overflow-hidden to ArtifactSurface article wrapper and chart
container div. Add min-w-0 to grid-item motion.div in DashboardPanel
to prevent CSS grid column blowout.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

### Spec coverage
- [x] Grid-item min-w-0 → Task 2
- [x] Card overflow-hidden → Task 1 (строка 393)
- [x] Chart div overflow-hidden + min-w-0 → Task 1 (строка 379)
- [x] Визуальная проверка → Task 3
- [x] Commit → Task 4

### Placeholder scan
Нет TBD / TODO / «реализовать позже».

### Type consistency
Только CSS-классы Tailwind — несовместимостей нет.
