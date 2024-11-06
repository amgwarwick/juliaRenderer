#include <EGL/egl.h>
#include <stdio.h>

static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
};

int setup_egl(int width, int height) {
    EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    if (eglDpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return -1;
    }

    EGLint major, minor;
    if (!eglInitialize(eglDpy, &major, &minor)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return -1;
    }

    EGLint numConfigs;
    EGLConfig eglCfg;
    if (!eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs)) {
        fprintf(stderr, "Failed to choose EGL config\n");
        return -1;
    }

    // Use width and height passed as parameters
    EGLint pbufferAttribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };

    EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    if (eglSurf == EGL_NO_SURFACE) {
        fprintf(stderr, "Failed to create EGL surface\n");
        return -1;
    }

    eglBindAPI(EGL_OPENGL_API);

    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
    if (eglCtx == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create EGL context\n");
        return -1;
    }

    if (!eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx)) {
        fprintf(stderr, "Failed to make EGL context current\n");
        return -1;
    }

    return 0;
}

