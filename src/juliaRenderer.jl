module juliaRenderer
using StaticArrays
import GLAbstraction
using ModernGL
using MuJoCo
using LinearAlgebra
import DataStructures: DefaultDict
import Images
include("basic_geoms.jl")
include("egl_utils.jl")
include("opengl_utils.jl")
include("geom_renderer.jl")
include("extract_data.jl")
include("batch_renderer.jl")
include("textures.jl")
export BatchRenderer, render
end
