module EGLUtils
export init_egl, free!

#I took these from the header of my egl.h; not sure how portable that is
#Someone should create proper EGL bindings for Julia
const EGL = "libEGL.so"
const EGL_DEFAULT_DISPLAY = 0
const EGL_NONE = 0x3038
const EGL_HEIGHT = 0x3056
const EGL_WIDTH = 0x3057
const EGL_SURFACE_TYPE = 0x3033
const EGL_PBUFFER_BIT = 0x01	
const EGL_BLUE_SIZE = 0x3022
const EGL_GREEN_SIZE = 0x3023
const EGL_RED_SIZE = 0x3024
const EGL_DEPTH_SIZE = 0x3025
const EGL_OPENGL_API = 0x30A2
const EGL_OPENGL_BIT = 0x0008
const EGL_RENDERABLE_TYPE = 0x3040
const EGL_CONTEXT_MAJOR_VERSION = 0x3098
const EGL_NO_CONTEXT = 0
const EGLDisplayType = Ptr{Nothing} #Void pointer

struct EGLResources
    eglDisplay::EGLDisplayType
    eglContext::Ptr{Nothing}
    eglSurface::Ptr{Nothing}
    eglMajor::Libc.Cint
    eglMinor::Libc.Cint
end

function init_egl(width::Integer, height::Integer)::EGLResources
    eglDisplay = ccall((:eglGetDisplay, EGL), EGLDisplayType, (Libc.Cint,), EGL_DEFAULT_DISPLAY)
    if eglDisplay == 0
        error("Failed to create EGL display")
    end

    egl_major = Ref{Libc.Cint}(0)
    egl_minor = Ref{Libc.Cint}(0)
    eglInitialized = ccall((:eglInitialize, EGL), Libc.Cint,
                           (EGLDisplayType, Ptr{Cint}, Ptr{Cint}),
                            eglDisplay,     egl_major, egl_minor)
    if eglInitialized != 1
        error("Failed to initialize EGL.")
    end

    # Create an attribute list for the configuration
    config_attribs = Libc.Cint[
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    ]

    config = Ref{Ptr{Nothing}}(C_NULL)
    num_configs = Ref{Cint}(0)
    # Load `eglChooseConfig`
    eglChooseConfig = ccall((:eglChooseConfig, EGL), Cint,
                            (EGLDisplayType, Ptr{Cint},      Ptr{Ptr{Nothing}}, Cint, Ptr{Cint}),
                            eglDisplay,  config_attribs, config,            1,    num_configs)
    if eglChooseConfig != 1 || num_configs[] == 0
        error("Failed to choose EGL configuration")
    end

    eglBindAPI = ccall((:eglBindAPI, EGL), Cint, (Cint,), EGL_OPENGL_API)
    if eglBindAPI != 1
        error("Failed to bind OpenGL API")
    end

    context_attribs = Libc.Cint[
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_NONE
    ]
    eglContext = ccall((:eglCreateContext, EGL), Ptr{Nothing}, 
                        (EGLDisplayType, Ptr{Nothing}, Libc.Cint, Ptr{Cint}),
                        eglDisplay, config[], EGL_NO_CONTEXT, context_attribs)
    if eglContext == 0
        error("Failed to create EGL context")
    end
    pbuffer_attribs = Libc.Cint[EGL_WIDTH, width,
                            EGL_HEIGHT, height,
                            EGL_NONE]

    eglSurface = ccall((:eglCreatePbufferSurface, EGL), Ptr{Nothing},
                        (EGLDisplayType, Ptr{Nothing}, Ptr{Cint}),
                        eglDisplay, config[], pbuffer_attribs)

    eglCurrent = ccall((:eglMakeCurrent, EGL), Libc.Cint, (EGLDisplayType, Ptr{Nothing}, Ptr{Nothing}, Ptr{Nothing}),
                                            eglDisplay, eglSurface, eglSurface, eglContext)
    if eglCurrent != 1
        error("Failed to make EGL context current")
    end
    return EGLResources(eglDisplay, eglContext, eglSurface, egl_major[], egl_minor[])
end

function free!(eglResources::EGLResources)
    eglDisplay = eglResources.eglDisplay
    eglContext = eglResources.eglContext
    eglSurface = eglResources.eglSurface
    ccall((:eglDestroyContext, EGL), Cint, (EGLDisplayType, Ptr{Nothing}), eglDisplay, eglContext)
    ccall((:eglDestroySurface, EGL), Cint, (EGLDisplayType, Ptr{Nothing}), eglDisplay, eglSurface)
    ccall((:eglTerminate, EGL), Cint, (EGLDisplayType,), eglDisplay)
end

function check_lib_egl()
    try
        Libdl.dlopen(EGL)
    catch e
        println("$EGL not found. Please make sure your system has EGL installed.")
        error(e)
    end
end

end