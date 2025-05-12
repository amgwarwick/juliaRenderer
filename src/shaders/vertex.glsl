#version 330 core

layout(location = 0) in vec3 aPos;     
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in mat4 modelMatrix;
layout(location = 7) in vec4 rgba;
layout(location = 8) in float offset;
layout(location = 9) in float tex_id;

out vec4 vColor;
out vec3 Normal;
out vec3 FragPos;
out float left;
out float right;
out vec3 texCoords;
out float n_env_frag;
out float res_frag;

uniform mat4 view;        
uniform mat4 projection;

uniform float n_env;
uniform float res;

void main()
{   

    left = -1 + 2*offset/n_env;
    right = -1 + 2*(offset+1)/n_env;
    float avg = (left + right)/2;
    
    mat4 envScale = mat4(1/n_env, 0, 0, 0, 
                            0, 1, 0, 0, 
                            0, 0, 1, 0, 
                            0, 0, 0, 1);
                            
    mat4 envTranslate = mat4(1, 0, 0, 0, 
                                0, 1, 0, 0, 
                                0, 0, 1, 0, 
                                avg, 0, 0, 1);
                                
    vec4 pos = projection * view * modelMatrix * vec4(aPos, 1.0);
    gl_Position = envTranslate * envScale * pos;
    vColor = rgba;
    Normal = normalize(mat3(transpose(inverse(modelMatrix))) * normalize(vec3(aNormal)));
    FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
    texCoords = vec3(aTexCoords, tex_id);
    n_env_frag = n_env;
    res_frag = res;
}
