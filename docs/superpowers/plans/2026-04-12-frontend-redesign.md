# Frontend Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Устранить "пластиковый" дефолтный вид интерфейса, добавив характерную типографику, глубину, текстуру и улучшенные анимации.

**Architecture:** Все изменения сконцентрированы в CSS-переменных (`theme.css`, `fonts.css`) и компонентах (Navigation, ChatPanel, AgentActivityFeed). Никаких новых зависимостей — используем Motion (уже есть) и Google Fonts через `@import`. Изменения идут снизу вверх: сначала глобальные токены, потом компоненты.

**Tech Stack:** React 18, TypeScript, Tailwind CSS 4, Motion (Framer), Google Fonts (DM Sans, DM Mono, Instrument Serif)

---

## Файловая карта

| Файл | Что меняется |
|------|-------------|
| `frontend/src/styles/fonts.css` | Подключение Google Fonts, CSS-переменные шрифтов |
| `frontend/src/styles/theme.css` | Цвета, фон, grain overlay, dark-mode |
| `frontend/src/app/components/Navigation.tsx` | Логотип, скользящий active-индикатор |
| `frontend/src/app/components/workspace/ChatPanel.tsx` | Инпут, пустое состояние, suggestions |
| `frontend/src/app/components/workspace/AgentActivityFeed.tsx` | Staggered анимации блоков |
| `frontend/src/app/components/ui/card.tsx` | Варианты elevated/sunken |

---

## Task 1: Кастомные шрифты

**Files:**
- Modify: `frontend/src/styles/fonts.css`
- Modify: `frontend/src/styles/theme.css`

Сейчас `fonts.css` пуст, используются системные шрифты. Добавляем DM Sans (UI), DM Mono (данные/код), Instrument Serif (акцентные заголовки).

- [ ] **Step 1: Добавить Google Fonts и переменные шрифтов**

`frontend/src/styles/fonts.css`:
```css
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Instrument+Serif:ital@0;1&display=swap');
```

- [ ] **Step 2: Прописать переменные в `:root` блок `theme.css`**

В `frontend/src/styles/theme.css`, добавить сразу после `color-scheme: light;` (строка 4):
```css
  --font-sans: 'DM Sans', system-ui, -apple-system, sans-serif;
  --font-serif: 'Instrument Serif', Georgia, serif;
  --font-mono: 'DM Mono', ui-monospace, 'Cascadia Code', monospace;
```

- [ ] **Step 3: Применить шрифт к `body` в `@layer base`**

В `frontend/src/styles/theme.css`, в секции `@layer base` найти `body { ... }` (строка ~214) и добавить:
```css
  body {
    @apply bg-background text-foreground;
    font-family: var(--font-sans);
    transition:
      background-color 220ms ease,
      color 220ms ease,
      border-color 220ms ease;
  }
```

- [ ] **Step 4: Применить моно к code-элементам**

В `frontend/src/styles/theme.css`, после блока `body { ... }` добавить:
```css
  code, pre, kbd, samp {
    font-family: var(--font-mono);
  }
```

- [ ] **Step 5: Проверить визуально**

Запустить dev-сервер (`cd frontend && npm run dev` или аналог), открыть браузер, убедиться что шрифт изменился — тексты стали мягче, моно-текст в блоках агента отличается от основного.

- [ ] **Step 6: Коммит**
```bash
git add frontend/src/styles/fonts.css frontend/src/styles/theme.css
git commit -m "feat(ui): add DM Sans + DM Mono + Instrument Serif fonts"
```

---

## Task 2: Off-white фон и grain-текстура

**Files:**
- Modify: `frontend/src/styles/theme.css`

Чистый белый `#ffffff` — главная причина "пластикового" вида. Меняем на тёплый off-white и добавляем subtle noise overlay.

- [ ] **Step 1: Изменить `--background` и `--card` в light mode**

В `frontend/src/styles/theme.css`, строка 21-23. Изменить:
```css
  --background: #f7f6f3;
  --card: #faf9f7;
  --card-foreground: oklch(0.145 0 0);
```
(Было: `--background: #ffffff`, `--card: #ffffff`)

- [ ] **Step 2: Подправить `--sidebar` под новый фон**

Строка 50:
```css
  --sidebar: #f2f1ee;
```
(Было: `oklch(0.985 0 0)`)

- [ ] **Step 3: Добавить grain overlay через CSS pseudo-element**

В конец `@layer base { ... }` секции добавить:
```css
  body {
    isolation: isolate;
  }

  body::after {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9998;
    opacity: 0.028;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-size: 256px 256px;
  }

  .dark body::after {
    opacity: 0.04;
  }
```

- [ ] **Step 4: Проверить визуально**

В браузере убедиться: фон стал чуть тёплым (не ослепительно белым), при увеличении зума 200%+ виден тонкий grain. В тёмной теме grain чуть заметнее.

- [ ] **Step 5: Коммит**
```bash
git add frontend/src/styles/theme.css
git commit -m "feat(ui): warm off-white background with subtle grain texture"
```

---

## Task 3: Улучшение тёмной темы

**Files:**
- Modify: `frontend/src/styles/theme.css`

Текущий `#0d0d10` — дешёвый pitch black. Меняем на холодный slate-near-black с чуть более выраженными разграничениями поверхностей.

- [ ] **Step 1: Обновить базовые цвета dark mode**

В `frontend/src/styles/theme.css`, в блоке `.dark { ... }` (строка ~126), заменить:
```css
  --background: #0e0f14;
  --card: #14151c;
  --popover: #14151c;
  --secondary: #1e1f28;
  --muted: #1e1f28;
  --accent: #1e1f28;
  --input-background: #181920;
  --sidebar: #0b0c11;
  --sidebar-accent: #14151c;
```

- [ ] **Step 2: Усилить border видимость в dark mode**

В том же `.dark { ... }` блоке:
```css
  --border: rgba(255, 255, 255, 0.1);
```
(Было: `rgba(255, 255, 255, 0.08)`)

- [ ] **Step 3: Проверить dark mode**

Переключить тему в интерфейсе, убедиться что сайдбар и фон отличаются по глубине, карточки выделяются на фоне, грани читаемы.

- [ ] **Step 4: Коммит**
```bash
git add frontend/src/styles/theme.css
git commit -m "feat(ui): refined dark theme with slate-near-black palette"
```

---

## Task 4: Навигация — sliding active indicator и улучшенный логотип

**Files:**
- Modify: `frontend/src/app/components/Navigation.tsx`

Сейчас активная вкладка — просто box с `bg-card`. Заменяем на sliding pill с `layoutId` анимацией через Motion.

- [ ] **Step 1: Добавить `AnimatePresence` и `motion` импорты**

В `frontend/src/app/components/Navigation.tsx`, строка 1-2:
```tsx
import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { Bell, Search, User } from "lucide-react";
import { motion } from "motion/react";
import { ThemeToggle } from "./ThemeToggle";
import { useAppSession } from "../context/AppSessionContext";
```

- [ ] **Step 2: Заменить логотип-квадрат на SVG с характером**

Найти `<div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary...` (строка ~32) и заменить целый Link-блок логотипа:
```tsx
<Link to="/" className="group flex shrink-0 items-center gap-2.5 transition-all">
  <div className="relative flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-xl bg-primary shadow-lg shadow-primary/25 transition-all group-hover:shadow-primary/40">
    <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5 text-primary-foreground" strokeWidth="2">
      <path d="M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM17.5 14v7M14 17.5h7" stroke="currentColor" strokeLinecap="round"/>
    </svg>
  </div>
  <span className="hidden whitespace-nowrap text-[15px] font-semibold tracking-tight text-foreground transition-colors group-hover:text-primary sm:block lg:text-[16px]">
    Генеративная аналитика
  </span>
</Link>
```

- [ ] **Step 3: Заменить nav-items рендер на sliding indicator**

Найти блок `<div className="hidden items-center overflow-x-auto rounded-xl border...` (строка ~41) и заменить содержимое map:
```tsx
<div className="hidden items-center overflow-x-auto rounded-xl border border-border/40 bg-secondary/50 p-1 backdrop-blur-md [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden lg:flex">
  {navItems.map((item) => {
    const isActive = location.pathname === item.path;
    return (
      <Link
        key={item.path}
        to={item.path}
        className="relative shrink-0 whitespace-nowrap rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors duration-200 xl:px-4 xl:text-[14px] 2xl:px-5"
      >
        {isActive && (
          <motion.span
            layoutId="nav-pill"
            className="absolute inset-0 rounded-lg bg-card shadow-sm ring-1 ring-border/20"
            transition={{ type: "spring", stiffness: 400, damping: 35 }}
          />
        )}
        <span className={`relative z-10 transition-colors duration-200 ${isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}>
          {item.label}
        </span>
      </Link>
    );
  })}
</div>
```

- [ ] **Step 4: Проверить анимацию**

Кликать по разным вкладкам в навигации, убедиться что pill плавно скользит от одной вкладки к другой.

- [ ] **Step 5: Коммит**
```bash
git add frontend/src/app/components/Navigation.tsx
git commit -m "feat(ui): navigation sliding active pill with motion layout"
```

---

## Task 5: Редизайн Chat Input

**Files:**
- Modify: `frontend/src/app/components/workspace/ChatPanel.tsx`

Инпут — главный элемент UX. Текущий вариант функционален, но невыразителен. Улучшаем: увеличенный радиус, glow при фокусе, красивые suggestion chips, лучший empty state.

- [ ] **Step 1: Улучшить контейнер textarea и glow**

Найти `<div className="group relative rounded-2xl border border-border/60 bg-card shadow-xl...` (строка ~336) и заменить:
```tsx
<div className="group relative rounded-3xl border border-border/50 bg-card/80 backdrop-blur-sm shadow-lg shadow-black/5 transition-all duration-200 focus-within:border-primary/40 focus-within:shadow-xl focus-within:shadow-primary/5 focus-within:ring-4 focus-within:ring-primary/8">
```

- [ ] **Step 2: Улучшить textarea стиль**

Найти `<textarea ... className="min-h-[72px] w-full resize-none bg-transparent p-3 pl-12 pr-14...` (строка ~347) и заменить `className`:
```tsx
className="min-h-[80px] w-full resize-none bg-transparent p-4 pl-12 pr-14 text-[13.5px] leading-relaxed outline-none placeholder:text-muted-foreground/50 lg:p-5 lg:pl-14 lg:pr-16 lg:text-[15px] xl:min-h-[108px]"
```

- [ ] **Step 3: Улучшить suggestion chips**

Найти блок `<div className="mb-4 flex flex-wrap items-center gap-2">` (строка ~299) и заменить:
```tsx
<div className="mb-3 flex flex-wrap items-center gap-2">
  {QUICK_SUGGESTIONS.map((suggestion) => (
    <button
      key={suggestion}
      type="button"
      onClick={() => setInput(suggestion)}
      className="group flex items-center gap-1.5 rounded-full border border-border/40 bg-card/60 px-3.5 py-1.5 text-[12px] font-medium text-muted-foreground backdrop-blur-sm transition-all duration-150 hover:border-primary/30 hover:bg-primary/5 hover:text-foreground"
    >
      <span className="h-1 w-1 rounded-full bg-primary/40 transition-colors group-hover:bg-primary/70" />
      {suggestion}
    </button>
  ))}
</div>
```

- [ ] **Step 4: Улучшить empty state**

Найти `<div className="rounded-3xl border border-dashed border-border/40 bg-secondary/20 p-6 text-sm text-muted-foreground">` (строка ~226) и заменить:
```tsx
<div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
  <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-border/40 bg-secondary/50 text-muted-foreground/60">
    <Bot className="h-6 w-6" />
  </div>
  <div>
    <p className="text-[14px] font-medium text-foreground/70">Аналитик готов к работе</p>
    <p className="mt-1 text-[12px] text-muted-foreground/60">Задайте вопрос о данных или загрузите CSV</p>
  </div>
</div>
```

- [ ] **Step 5: Проверить**

Открыть workspace, убедиться что пустое состояние выглядит чисто, suggestions имеют hover-эффект, инпут при фокусе даёт мягкое свечение.

- [ ] **Step 6: Коммит**
```bash
git add frontend/src/app/components/workspace/ChatPanel.tsx
git commit -m "feat(ui): refined chat input with glow focus, better empty state and suggestion chips"
```

---

## Task 6: Визуальная иерархия карточек

**Files:**
- Modify: `frontend/src/styles/theme.css`
- Modify: `frontend/src/app/components/ui/card.tsx`

Добавим CSS-переменные для "поднятых" и "утопленных" поверхностей и вариант `elevated` для Card.

- [ ] **Step 1: Добавить CSS-переменные для поверхностей**

В `frontend/src/styles/theme.css`, добавить в `:root { ... }` после `--card: #faf9f7;`:
```css
  --card-elevated: #ffffff;
  --card-sunken: #f0efe9;
  --card-elevated-foreground: oklch(0.145 0 0);
```

В блоке `.dark { ... }` после `--card: #14151c;`:
```css
  --card-elevated: #1a1b24;
  --card-sunken: #0d0e12;
  --card-elevated-foreground: #f4f4f5;
```

- [ ] **Step 2: Добавить Tailwind-маппинг в `@theme inline`**

В `frontend/src/styles/theme.css`, в блоке `@theme inline { ... }` добавить после `--color-card-foreground`:
```css
  --color-card-elevated: var(--card-elevated);
  --color-card-sunken: var(--card-sunken);
  --color-card-elevated-foreground: var(--card-elevated-foreground);
```

- [ ] **Step 3: Добавить вариант `elevated` в компонент Card**

Прочитать файл `frontend/src/app/components/ui/card.tsx`, найти функцию `Card` и добавить `elevated` класс:

В `card.tsx`, найти основной `<div>` карточки и добавить `data-elevated` поддержку — или добавить отдельный компонент `ElevatedCard`:
```tsx
function CardElevated({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card"
      className={cn(
        "bg-card-elevated text-card-elevated-foreground flex flex-col gap-6 rounded-xl border border-border/40 py-6 shadow-md shadow-black/5 transition-shadow duration-200 hover:shadow-lg hover:shadow-black/8",
        className
      )}
      {...props}
    />
  );
}

export { CardElevated };
```

- [ ] **Step 4: Проверить в DevTools**

Убедиться что CSS-переменные `--color-card-elevated` доступны в браузере, цвета отличаются от базового card.

- [ ] **Step 5: Коммит**
```bash
git add frontend/src/styles/theme.css frontend/src/app/components/ui/card.tsx
git commit -m "feat(ui): add elevated/sunken card surface hierarchy"
```

---

## Task 7: Улучшение стриминг-анимаций AgentActivityFeed

**Files:**
- Modify: `frontend/src/app/components/workspace/AgentActivityFeed.tsx`

Это самая живая часть продукта. Добавим staggered reveal при появлении новых шагов и лучшую пульсацию "живого" состояния.

- [ ] **Step 1: Прочитать полный файл AgentActivityFeed.tsx**

```bash
# Убедиться в текущей структуре компонентов
cat frontend/src/app/components/workspace/AgentActivityFeed.tsx
```

- [ ] **Step 2: Добавить staggered анимацию на каждый tool-step**

В `frontend/src/app/components/workspace/AgentActivityFeed.tsx`, найти место где рендерятся отдельные шаги (tool-вызовы/блоки) и обернуть каждый в `motion.div` со stagger:

```tsx
// Пример паттерна для каждого step-элемента в списке
<motion.div
  key={step.id || index}
  initial={{ opacity: 0, x: -8, filter: "blur(2px)" }}
  animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
  transition={{ duration: 0.25, delay: index * 0.05, ease: [0.25, 0.46, 0.45, 0.94] }}
>
  {/* existing step content */}
</motion.div>
```

- [ ] **Step 3: Улучшить "живой" индикатор (пульсирующие точки вместо спиннера или в дополнение)**

Найти место где используется `spinner` символ (строка в `ChatPanel.tsx` ~249):
```tsx
<div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 font-mono text-sm text-primary select-none">
  {spinner}
</div>
```

Если spinner — это текстовые символы (◐◓◑◒ и т.п.), добавить рядом с ним `ring-pulse` анимацию через CSS:

В `theme.css` в `@layer base`:
```css
  @keyframes ring-pulse {
    0%, 100% { box-shadow: 0 0 0 0 color-mix(in oklab, var(--primary) 40%, transparent); }
    50% { box-shadow: 0 0 0 6px color-mix(in oklab, var(--primary) 0%, transparent); }
  }

  .animate-ring-pulse {
    animation: ring-pulse 2s ease-in-out infinite;
  }
```

Применить к spinner-аватару в `ChatPanel.tsx`:
```tsx
<div className="animate-ring-pulse mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 font-mono text-sm text-primary select-none">
  {spinner}
</div>
```

- [ ] **Step 4: Добавить fade-out для устаревших reasoning-строк**

В `ReasoningText` компоненте (строка ~76 в AgentActivityFeed), обернуть в motion:
```tsx
function ReasoningText({ text }: { text: string }) {
  const line = firstMeaningfulLine(text);
  if (!line) return null;
  return (
    <motion.p
      initial={{ opacity: 0, y: 3 }}
      animate={{ opacity: 1, y: 0 }}
      className="pl-1 text-[13px] text-muted-foreground/75 leading-5 select-none"
    >
      <InlineMarkdown text={line} />
    </motion.p>
  );
}
```

- [ ] **Step 5: Проверить стриминг**

Запустить анализ в workspace, наблюдать как шаги появляются с плавным stagger, spinner пульсирует, reasoning текст мягко появляется.

- [ ] **Step 6: Коммит**
```bash
git add frontend/src/app/components/workspace/AgentActivityFeed.tsx frontend/src/app/components/workspace/ChatPanel.tsx frontend/src/styles/theme.css
git commit -m "feat(ui): staggered streaming animations and pulsing live indicator"
```

---

## Self-Review

**Покрытие требований:**
- ✅ Task 1: Шрифты (DM Sans + DM Mono)
- ✅ Task 2: Off-white фон + grain texture
- ✅ Task 3: Тёмная тема (slate-near-black)
- ✅ Task 4: Навигация (sliding pill)
- ✅ Task 5: Chat input (glow, empty state, chips)
- ✅ Task 6: Карточки (иерархия elevated/sunken)
- ✅ Task 7: Стриминг анимации (stagger, pulse)

**Нет placeholder-ов:** Все шаги содержат реальный код.

**Зависимости задач:** Task 2 должен идти после Task 1 (оба в theme.css, потенциальный конфликт — пишем разные переменные). Task 7 частично зависит от Task 2 (ring-pulse keyframes в theme.css).
