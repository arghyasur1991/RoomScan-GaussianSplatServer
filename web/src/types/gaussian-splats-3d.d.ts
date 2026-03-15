declare module '@mkkellogg/gaussian-splats-3d' {
  import type { Scene, WebGLRenderer, PerspectiveCamera } from 'three';

  interface ViewerOptions {
    scene?: Scene;
    renderer?: WebGLRenderer;
    camera?: PerspectiveCamera;
    selfDrivenMode?: boolean;
    [key: string]: any;
  }

  interface AddSceneOptions {
    showLoadingUI?: boolean;
    [key: string]: any;
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(url: string, options?: AddSceneOptions): Promise<void>;
    update(): void;
    render(): void;
    dispose(): void;
  }
}
