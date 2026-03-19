
import { createRoot } from "react-dom/client";
import App from "./app/App.tsx";
import { ThemeProvider } from "./app/components/ThemeProvider.tsx";
import { AppSessionProvider } from "./app/context/AppSessionContext.tsx";
import { ChatAgentProvider } from "./app/context/ChatAgentContext.tsx";
import "./styles/index.css";

createRoot(document.getElementById("root")!).render(
  <ThemeProvider>
    <AppSessionProvider>
      <ChatAgentProvider>
        <App />
      </ChatAgentProvider>
    </AppSessionProvider>
  </ThemeProvider>,
);
  
