import { createElement } from "react";
import { Navigate, createBrowserRouter } from "react-router";
import { Platform } from "./pages/Platform";
import { Auth } from "./pages/Auth";
import { Workspace } from "./pages/Workspace";
import { Technical } from "./pages/Technical";
import { Phoenix } from "./pages/Phoenix";
import { Sessions } from "./pages/Sessions";
import { Account } from "./pages/Account";

function RedirectToAuth() {
  return createElement(Navigate, { to: "/auth", replace: true });
}

function RedirectToWorkspace() {
  return createElement(Navigate, { to: "/workspace", replace: true });
}

const routerBasename = import.meta.env.BASE_URL.replace(/\/$/, "") || undefined;

export const router = createBrowserRouter(
  [
    {
      path: "/",
      Component: Platform,
    },
    {
      path: "/auth",
      Component: Auth,
    },
    {
      path: "/user",
      Component: RedirectToAuth,
    },
    {
      path: "/account",
      Component: Account,
    },
    {
      path: "/workspace",
      Component: Workspace,
    },
    {
      path: "/app",
      Component: RedirectToWorkspace,
    },
    {
      path: "/technical",
      Component: Technical,
    },
    {
      path: "/tracing",
      Component: Phoenix,
    },
    {
      path: "/sessions",
      Component: Sessions,
    },
  ],
  {
    basename: routerBasename,
  },
);
