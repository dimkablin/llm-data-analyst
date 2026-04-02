import { Navigation } from "../components/Navigation";
import { Link } from "react-router";
import { useEffect, useState } from "react";
import {
  MessageSquare,
  Upload,
  BarChart3,
  Table2,
  Activity,
  Settings,
  ChevronRight,
  Zap,
  Shield,
  Sparkles,
  Workflow,
  LayoutDashboard,
  ArrowUpRight,
  Mouse,
  ChevronDown,
} from "lucide-react";
import { motion } from "motion/react";

const features = [
  {
    icon: MessageSquare,
    title: "Аналитический чат",
    description:
      "Вопросы на естественном языке превращаются в объяснимый аналитический ответ, а не в абстрактный текст без опоры на данные.",
    color: "text-blue-400",
    bg: "bg-blue-400/10",
  },
  {
    icon: Upload,
    title: "Загрузка CSV",
    description:
      "Данные подключаются прямо в сессию. Команда может быстро запустить пилот, демо или разбор конкретного кейса без долгой подготовки интерфейса.",
    color: "text-violet-400",
    bg: "bg-violet-400/10",
  },
  {
    icon: Workflow,
    title: "Агентный анализ",
    description:
      "Платформа строит шаги анализа, вызывает инструменты и собирает результат в воспроизводимый сценарий с контролируемым loop-процессом.",
    color: "text-cyan-400",
    bg: "bg-cyan-400/10",
  },
  {
    icon: BarChart3,
    title: "Графики и визуализации",
    description:
      "Из запроса рождаются не только выводы, но и готовые артефакты: графики, сравнения, тренды и визуальные акценты для обсуждения с заказчиком.",
    color: "text-emerald-400",
    bg: "bg-emerald-400/10",
  },
  {
    icon: Table2,
    title: "Таблицы и метрики",
    description:
      "Система возвращает структурированные KPI, таблицы и числовые срезы, которые легко проверить, закрепить в дашборде или вынести в отчет.",
    color: "text-amber-400",
    bg: "bg-amber-400/10",
  },
  {
    icon: LayoutDashboard,
    title: "Живой dashboard",
    description:
      "Ключевые артефакты можно собирать в визуальный слой прямо по ходу диалога: от first-pass анализа до презентации результата команде.",
    color: "text-rose-400",
    bg: "bg-rose-400/10",
  },
];

const benefits = [
  {
    title: "Ускорение принятия решений",
    text: "Первый обоснованный ответ появляется в рамках одной сессии, без длинной очереди к BI-команде и потери контекста между созвонами.",
  },
  {
    title: "Прозрачность и доверие",
    text: "Каждый вывод можно подтвердить артефактами: таблицей, графиком, значением и логикой построения ответа.",
  },
  {
    title: "Снижение операционной нагрузки",
    text: "Типовые аналитические вопросы переходят в self-service сценарий и освобождают команду данных для более сложных задач.",
  },
  {
    title: "Production-ready контур",
    text: "Аутентификация, изоляция сессий, лимиты датасета, SSE-стриминг и наблюдаемость позволяют запускать платформу не только в демо, но и в пилоте.",
  },
];

const useCases = [
  {
    title: "Выручка и маржинальность",
    text: "Быстро находите драйверы роста и просадки по периодам, сегментам и продуктам без ручной сборки десятков отдельных разрезов.",
  },
  {
    title: "Продуктовая аналитика",
    text: "Проверяйте гипотезы по воронке, retention и поведению пользователей в формате «вопрос → ответ → артефакт».",
  },
  {
    title: "Операционная эффективность",
    text: "Выявляйте узкие места в процессах и быстрее готовьте обоснование для управленческих решений.",
  },
];

const flow = [
  {
    step: "01",
    title: "Загрузка данных",
    text: "Подключаете CSV в сессию и сразу переходите к работе с вопросами бизнеса.",
  },
  {
    step: "02",
    title: "Агентный разбор",
    text: "Система планирует анализ, выполняет шаги и формирует проверяемые артефакты.",
  },
  {
    step: "03",
    title: "Сборка визуального слоя",
    text: "Ключевые графики, таблицы и KPI закрепляются в dashboard без разрыва сценария.",
  },
  {
    step: "04",
    title: "Принятие решения",
    text: "Результат готов к презентации, обсуждению с командой и следующему шагу в пилоте.",
  },
];

const proofMetrics = [
  { value: "5-15 мин", label: "до первого содержательного ответа" },
  { value: "100 MB", label: "лимит датасета по backend-контуру" },
  { value: "SSE + Agent Loop", label: "живой поток ответа и управляемый анализ" },
];

export function Platform() {
  const [showScrollCue, setShowScrollCue] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      setShowScrollCue(window.scrollY < 80);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30">
      <Navigation />

      <section className="relative flex items-center overflow-hidden border-b border-border/40 pt-28 pb-14 xl:pt-32 xl:pb-16" style={{ minHeight: "calc((100vh - 72px) / var(--ui-zoom, 1))" }}>
        <div className="absolute left-1/2 top-0 -z-10 h-[680px] w-[1100px] -translate-x-1/2 rounded-full bg-primary/10 blur-[140px] opacity-40" />
        <div className="absolute right-[12%] top-24 -z-10 h-56 w-56 rounded-full bg-blue-400/10 blur-[100px]" />
        <div className="absolute left-[10%] bottom-16 -z-10 h-44 w-44 rounded-full bg-emerald-400/10 blur-[90px]" />
        <div className="max-w-[1240px] mx-auto px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center max-w-5xl mx-auto"
          >
            <h1 className="mb-7 text-5xl font-bold leading-[1.02] tracking-tight sm:text-6xl md:text-7xl xl:text-7xl 2xl:text-[88px]">
              Генеративная{" "}
              <span className="bg-gradient-to-r from-primary via-blue-400 to-cyan-300 bg-clip-text text-transparent">
                аналитика
              </span>
            </h1>

            <p className="mx-auto mb-10 max-w-3xl text-lg leading-relaxed text-muted-foreground md:text-xl 2xl:text-[22px]">
              Платформа, которая объединяет аналитический чат, агентный анализ и dashboard в одном
              интерфейсе. Из сырых данных она собирает понятный, визуальный и проверяемый ответ для
              бизнеса, продукта и presale-демо.
            </p>

            <div className="mb-12 flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Link
                to="/workspace"
                className="group relative flex items-center gap-2 overflow-hidden rounded-2xl bg-primary px-10 py-4.5 font-bold text-primary-foreground shadow-2xl shadow-primary/20 transition-all hover:shadow-primary/40 active:scale-95"
              >
                <div className="absolute inset-0 translate-y-full bg-white/10 transition-transform duration-300 group-hover:translate-y-0" />
                <Zap className="h-5 w-5 fill-current" />
                <span>Открыть рабочую область</span>
              </Link>
              <Link
                to="/technical"
                className="flex items-center gap-2 rounded-2xl border border-border/50 bg-secondary px-10 py-4.5 font-bold text-foreground transition-all hover:bg-muted active:scale-95"
              >
                <span>Технический контур</span>
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              </Link>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              {proofMetrics.map((metric, index) => (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.08 + index * 0.08 }}
                  className="rounded-[28px] border border-border/50 bg-card/70 p-6 text-left shadow-[0_14px_40px_rgba(15,23,42,0.04)] backdrop-blur-md"
                >
                  <div className="mb-2 text-3xl font-bold tracking-tight">{metric.value}</div>
                  <p className="text-sm leading-relaxed text-muted-foreground">{metric.label}</p>
                </motion.div>
              ))}
            </div>

          </motion.div>
        </div>

        <motion.a
          href="#platform-capabilities"
          initial={{ opacity: 0, y: 10 }}
          animate={{
            opacity: showScrollCue ? 1 : 0,
            y: showScrollCue ? [0, 6, 0] : 10,
          }}
          transition={{
            opacity: { duration: 0.25 },
            y: { delay: 0.7, duration: 1.8, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
          }}
          aria-hidden={!showScrollCue}
          className="fixed bottom-8 left-1/2 z-40 inline-flex -translate-x-1/2 flex-col items-center gap-2 text-muted-foreground transition-colors hover:text-foreground"
          style={{ pointerEvents: showScrollCue ? "auto" : "none" }}
        >
          <div className="flex items-center gap-2 rounded-full border border-border/50 bg-background/60 px-4 py-2 backdrop-blur-md">
            <Mouse className="h-4 w-4" />
            <span className="text-[12px] font-semibold uppercase tracking-[0.18em]">Обзор платформы</span>
          </div>
          <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border/40 bg-card/70">
            <ChevronDown className="h-5 w-5" />
          </div>
        </motion.a>
      </section>

      <section id="platform-capabilities" className="max-w-[1400px] mx-auto px-8 py-28">
        <div className="mb-20 text-center">
          <h2 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">Возможности платформы</h2>
          <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
            Все, что нужно для современного AI-анализа данных: от первой загрузки до визуального
            результата, который можно показывать клиенту уже на той же встрече.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 xl:grid-cols-3">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.08 }}
              className="group rounded-[32px] border border-border/50 bg-card p-10 transition-all hover:border-primary/30 hover:shadow-2xl hover:shadow-primary/5"
            >
              <div className={`mb-8 flex h-14 w-14 items-center justify-center rounded-2xl ${feature.bg} transition-transform group-hover:scale-110`}>
                <feature.icon className={`h-7 w-7 ${feature.color}`} />
              </div>
              <h3 className="mb-4 text-2xl font-bold transition-colors group-hover:text-primary">{feature.title}</h3>
              <p className="text-[16px] leading-relaxed text-muted-foreground">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="border-y border-border/40 bg-secondary/25 py-28">
        <div className="max-w-[1320px] mx-auto px-8">
          <div className="mb-16 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border/50 bg-background/70 px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">
                Почему это работает
              </div>
              <h2 className="text-4xl font-bold tracking-tight md:text-5xl">Что получает бизнес</h2>
            </div>
            <p className="max-w-xl text-[16px] leading-relaxed text-muted-foreground">
              Быстрый путь от вопроса к объяснимому ответу, который можно обсуждать с командой,
              показывать заказчику и использовать как основу для следующего решения.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {benefits.map((item, index) => (
              <motion.article
                key={item.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.06 }}
                className="rounded-[30px] border border-border/50 bg-card/80 p-8 shadow-sm"
              >
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                    <ArrowUpRight className="h-5 w-5" />
                  </div>
                  <h3 className="text-2xl font-bold tracking-tight">{item.title}</h3>
                </div>
                <p className="text-[15px] leading-relaxed text-muted-foreground">{item.text}</p>
              </motion.article>
            ))}
          </div>
        </div>
      </section>

      <section className="max-w-[1320px] mx-auto px-8 py-28">
        <div className="mb-16 max-w-2xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-border/50 bg-secondary/60 px-3 py-1.5 text-[11px] font-bold uppercase tracking-[0.24em] text-muted-foreground">
            Бизнес-сценарии
          </div>
          <h2 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">Где платформа дает быстрый эффект</h2>
          <p className="text-lg text-muted-foreground">
            Это не только красивый UI для демо. Это формат, который удобно продавать, пилотировать и
            затем переносить в регулярную аналитику.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          {useCases.map((item, index) => (
            <motion.article
              key={item.title}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.08 }}
              className="rounded-[32px] border border-border/50 bg-card p-8"
            >
              <h3 className="mb-4 text-2xl font-bold">{item.title}</h3>
              <p className="text-[15px] leading-relaxed text-muted-foreground">{item.text}</p>
            </motion.article>
          ))}
        </div>
      </section>

      <section className="relative overflow-hidden border-t border-border/40 py-28">
        <div className="absolute left-[10%] top-14 -z-10 h-52 w-52 rounded-full bg-primary/10 blur-[90px]" />
        <div className="max-w-[1320px] mx-auto px-8">
          <div className="mb-16 text-center">
            <h2 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">От вопроса к решению</h2>
            <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
              Прозрачный сценарий, который хорошо выглядит на демо и остается понятным в ежедневной
              эксплуатации.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
            {flow.map((item, index) => (
              <motion.article
                key={item.step}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.07 }}
                className="rounded-[30px] border border-border/50 bg-card/80 p-8 backdrop-blur-md"
              >
                <div className="mb-6 text-[13px] font-bold uppercase tracking-[0.24em] text-primary">{item.step}</div>
                <h3 className="mb-4 text-2xl font-bold tracking-tight">{item.title}</h3>
                <p className="text-[15px] leading-relaxed text-muted-foreground">{item.text}</p>
              </motion.article>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-background py-20">
        <div className="max-w-[1180px] mx-auto px-8">
          <div className="rounded-[40px] border border-border/50 bg-card px-8 py-12 text-center shadow-[0_20px_60px_rgba(15,23,42,0.06)] md:px-12">
            <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Shield className="h-7 w-7" />
            </div>
            <h2 className="mb-4 text-3xl font-bold tracking-tight md:text-4xl">Готово к запуску</h2>
            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Link
                to="/workspace"
                className="rounded-2xl bg-primary px-8 py-3.5 font-bold text-primary-foreground transition-all hover:opacity-95 active:scale-95"
              >
                Перейти в workspace
              </Link>
              <Link
                to="/technical"
                className="rounded-2xl border border-border/50 bg-secondary px-8 py-3.5 font-bold text-foreground transition-all hover:bg-muted active:scale-95"
              >
                Посмотреть технический контур
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
