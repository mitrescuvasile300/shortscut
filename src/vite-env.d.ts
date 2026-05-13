/// <reference types="vite/client" />

// Allow ?raw imports for any file type
declare module "*.py?raw" {
  const content: string;
  export default content;
}

interface ImportMetaEnv {
  readonly VITE_CONVEX_URL: string;
  readonly VITE_IS_PREVIEW: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
