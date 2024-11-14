module Renderer
using StaticArrays
using GLAbstraction
using ModernGL
using MuJoCo
using LinearAlgebra
import DataStructures: DefaultDict
include("basic_geoms.jl")
include("egl_utils.jl")
include("opengl_utils.jl")
include("geom_renderer.jl")
include("extract_data.jl")
include("batch_renderer.jl")
export BatchRenderer, render
end
