declare module 'tiff.js' {
    export default class Tiff {
        constructor(opts: { buffer: ArrayBuffer });
        toCanvas(): HTMLCanvasElement;
    }
}
