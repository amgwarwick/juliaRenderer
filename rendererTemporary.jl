using MuJoCo
using Libdl
using Base.Libc: Ptr
using LinearAlgebra
using GLM
using ModernGL, GLAbstraction
using FileIO
import Images
using ColorTypes
using LinearAlgebra
using GeometryTypes
using Colors
using ProgressMeter
using Profile

include("rendererStructureTemporary.jl")


n_env_cols = 4
n_env_rows = 4
model_path = "rodent_with_floor.xml"
model = MuJoCo.load_model(model_path)

# Create one data instance per environment cell.
datas = [MuJoCo.init_data(model) for _ = 1:(n_env_cols * n_env_rows)]
for data in datas
    MuJoCo.step!(model, data)
end

# Now call BatchRenderer with separate grid parameters:
batchRenderer = BatchRenderer(model, res=256, n_env_cols=n_env_cols, n_env_rows=n_env_rows)

function benchMark(batchRenderer, datas)
    for t = 1:32
        images = render(batchRenderer, datas)
    end
end

@time benchMark(batchRenderer, datas)


#=
using MuJoCo, HDF5, FileIO

# Parameters
n_envs = 4
n_env_cols = 2  # Adjusted for a 2x2 grid
n_env_rows = 2
model_path = "rodent_with_floor_scale080_edits.xml"
h5_file = "/home/utente/Downloads/diego_curated_snippets.h5"
output_dir = "demoVideo"
video_path = "output_video.mp4"

# Load model
model = MuJoCo.load_model(model_path)

# Create one data instance per environment
datas = [MuJoCo.init_data(model) for _ = 1:(n_env_cols * n_env_rows)]

# Create batch renderer
batchRenderer = BatchRenderer(model, res=1080, n_env_cols=n_env_cols, n_env_rows=n_env_rows)

# Ensure output directory exists
mkpath(output_dir)

# Open HDF5 file and process
h5open(h5_file, "r") do fid
    for t in 1:250
        for (j, data) in enumerate(datas)
            # Read HDF5 data for each timestep (t)
            pos_data = read(fid["clip_$(j)/walkers/walker_0/position"])
            quat_data = read(fid["clip_$(j)/walkers/walker_0/quaternion"])
            joint_data = read(fid["clip_$(j)/walkers/walker_0/joints"])

            # Extract the correct row (timestep t)
            data.qpos[1:3] .= pos_data[t, :]  # Position (1x3)
            data.qpos[4:7] .= quat_data[t, :] # Quaternion (1x4)
            data.qpos[8:end] .= joint_data[t, :]  # Joints (1x67)

            # Forward model step
            MuJoCo.forward!(model, data)
        end
        
        # Render image for the current timestep
        images = render(batchRenderer, datas)
        
        # Ensure images are valid before saving
        if !isempty(images)
            save(joinpath(output_dir, "frame_$(lpad(t, 3, '0')).png"), images)
        else
            println("Warning: No image data for timestep $t.")
        end
    end
end

# Convert frames to video using ffmpeg
run(`ffmpeg -framerate 30 -i $output_dir/frame_%03d.png -c:v libx264 -pix_fmt yuv420p $video_path`)
println("Video saved as $video_path")
=#

