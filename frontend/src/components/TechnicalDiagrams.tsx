type DiagramLabel = string | string[];

function ArrowDefs({ markerId }: { markerId: string }): JSX.Element {
  return (
    <defs>
      <marker id={markerId} markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 Z" fill="currentColor" />
      </marker>
    </defs>
  );
}

type DiagramShellProps = {
  children: JSX.Element;
  centered?: boolean;
};

function DiagramShell({ children, centered = false }: DiagramShellProps): JSX.Element {
  return <div className={`diagram-shell ${centered ? "diagram-centered" : ""}`}>{children}</div>;
}

function DiagramText({ x, y, label }: { x: number; y: number; label: DiagramLabel }): JSX.Element {
  const lines = Array.isArray(label) ? label : label.split("\n");
  const firstLineY = y - ((lines.length - 1) * 26) / 2 + 4;
  return (
    <text x={x} y={firstLineY} textAnchor="middle" className="diagram-text">
      {lines.map((line, index) => (
        <tspan key={`${line}-${index}`} x={x} dy={index === 0 ? 0 : 26}>
          {line}
        </tspan>
      ))}
    </text>
  );
}

function Arrow({ d, markerId }: { d: string; markerId: string }): JSX.Element {
  return <path d={d} className="diagram-line" markerEnd={`url(#${markerId})`} />;
}

function Node({
  x,
  y,
  w,
  h,
  label
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  label: DiagramLabel;
}): JSX.Element {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={14} ry={14} className="diagram-node" />
      <DiagramText x={x + w / 2} y={y + h / 2} label={label} />
    </g>
  );
}

function Decision({
  cx,
  cy,
  w,
  h,
  label
}: {
  cx: number;
  cy: number;
  w: number;
  h: number;
  label: DiagramLabel;
}): JSX.Element {
  const points = `${cx},${cy - h / 2} ${cx + w / 2},${cy} ${cx},${cy + h / 2} ${cx - w / 2},${cy}`;
  return (
    <g>
      <polygon points={points} className="diagram-decision" />
      <DiagramText x={cx} y={cy} label={label} />
    </g>
  );
}

export function ArchitectureStaticDiagram(): JSX.Element {
  const markerId = "arch-arrow";
  return (
    <DiagramShell>
      <svg viewBox="0 0 1740 700" role="img" aria-label="Архитектура системы">
        <ArrowDefs markerId={markerId} />

        <Node x={40} y={290} w={220} h={96} label={["Пользователь", "Браузер"]} />
        <Node x={330} y={290} w={220} h={96} label={["Фронтенд", "React UI"]} />
        <Node x={650} y={290} w={240} h={96} label={["FastAPI", "Бэкенд"]} />

        <Node x={980} y={70} w={220} h={96} label={["Auth БД", "SQLite"]} />
        <Node x={980} y={220} w={220} h={96} label={["Хранилище", "сессий"]} />
        <Node x={980} y={390} w={260} h={96} label={["Reason-Action", "Агент"]} />

        <Node x={1320} y={220} w={260} h={106} label={["Инструменты", "Pandas + Plotly"]} />
        <Node x={1320} y={420} w={220} h={96} label={["LLM", "Ollama / vLLM"]} />
        <Node x={1605} y={300} w={120} h={156} label={["Артефакты", "Вывод"]} />

        <Arrow d="M260,338 L330,338" markerId={markerId} />
        <Arrow d="M550,338 L650,338" markerId={markerId} />
        <Arrow d="M650,370 C610,405 585,405 550,370" markerId={markerId} />

        <Arrow d="M890,338 L930,338" markerId={markerId} />
        <Arrow d="M930,338 C948,300 952,150 980,118" markerId={markerId} />
        <Arrow d="M930,338 C958,338 952,268 980,268" markerId={markerId} />
        <Arrow d="M930,338 C948,372 954,438 980,438" markerId={markerId} />

        <Arrow d="M1240,438 C1305,438 1270,273 1320,273" markerId={markerId} />
        <Arrow d="M1240,438 C1285,438 1298,468 1320,468" markerId={markerId} />

        <Arrow d="M1580,273 C1614,276 1642,286 1665,300" markerId={markerId} />
        <Arrow d="M1540,468 C1602,468 1640,464 1665,456" markerId={markerId} />

        <Arrow d="M1320,468 C1120,620 860,600 770,386" markerId={markerId} />

        <text x="585" y="268" className="diagram-note">
          REST / SSE stream
        </text>
      </svg>
    </DiagramShell>
  );
}

export function AgentCycleStaticDiagram(): JSX.Element {
  const markerId = "agent-arrow";
  return (
    <DiagramShell>
      <svg viewBox="0 0 1680 560" role="img" aria-label="Reason-Action цикл">
        <ArrowDefs markerId={markerId} />

        <Node x={50} y={240} w={200} h={96} label={["Запрос", "пользователя"]} />
        <Node x={330} y={240} w={200} h={96} label={["План", "Шаг"]} />
        <Node x={610} y={240} w={230} h={96} label={["LLM", "Рассуждение"]} />

        <Node x={930} y={110} w={220} h={96} label={["Вызов", "инструмента"]} />
        <Node x={930} y={360} w={220} h={96} label={["Результат", "инструмента"]} />

        <Node x={1240} y={240} w={210} h={96} label={["Синтез", "Черновик ответа"]} />
        <Node x={1490} y={240} w={160} h={96} label={["Финальный", "ответ"]} />

        <Arrow d="M250,288 L330,288" markerId={markerId} />
        <Arrow d="M530,288 L610,288" markerId={markerId} />
        <Arrow d="M840,288 C900,288 875,158 930,158" markerId={markerId} />
        <Arrow d="M1040,206 L1040,360" markerId={markerId} />
        <Arrow d="M1150,408 C1220,408 1180,288 1240,288" markerId={markerId} />
        <Arrow d="M1450,288 L1490,288" markerId={markerId} />
        <Arrow d="M930,408 C760,540 660,520 725,336" markerId={markerId} />

        <text x="750" y="536" className="diagram-note">
          ReAct цикл до завершения
        </text>
      </svg>
    </DiagramShell>
  );
}

export function SecurityStaticDiagram(): JSX.Element {
  const markerId = "security-arrow";
  return (
    <DiagramShell>
      <svg viewBox="0 0 1680 520" role="img" aria-label="Безопасность и доступ">
        <ArrowDefs markerId={markerId} />

        <Node x={40} y={200} w={220} h={88} label="Вход" />
        <Node x={320} y={200} w={240} h={88} label={["Токен", "доступа"]} />
        <Decision cx={720} cy={244} w={250} h={190} label={["API", "запрос"]} />

        <Node x={860} y={90} w={260} h={92} label={["Пользователь", "определён"]} />
        <Node x={860} y={340} w={280} h={92} label={["401", "Не авторизован"]} />

        <Decision cx={1260} cy={136} w={250} h={180} label={["Владелец", "сессии?"]} />
        <Node x={1420} y={46} w={220} h={88} label={["Чтение / запись", "сессии"]} />
        <Node x={1420} y={180} w={220} h={88} label={["404", "Не найдено"]} />

        <Arrow d="M260,244 L320,244" markerId={markerId} />
        <Arrow d="M560,244 L595,244" markerId={markerId} />

        <Arrow d="M845,244 C920,244 955,210 990,182" markerId={markerId} />
        <Arrow d="M845,244 C920,244 955,305 1000,340" markerId={markerId} />

        <Arrow d="M1120,136 L1135,136" markerId={markerId} />
        <Arrow d="M1385,136 C1404,130 1412,104 1420,90" markerId={markerId} />
        <Arrow d="M1385,136 C1402,148 1412,198 1420,224" markerId={markerId} />

        <text x="900" y="220" className="diagram-note">
          Валидный
        </text>
        <text x="960" y="320" className="diagram-note">
          Невалидный
        </text>
        <text x="1432" y="36" className="diagram-note">
          Да
        </text>
        <text x="1432" y="296" className="diagram-note">
          Нет
        </text>
      </svg>
    </DiagramShell>
  );
}
