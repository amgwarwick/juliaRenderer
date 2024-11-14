#version 330 core

in vec4 vColor;
in float left;
in float right;
in vec3 Normal;
in vec3 FragPos;
in vec2 texCoords;
in float texID;
in float n_env_frag;
in float res_frag;

out vec4 FragColor;

uniform vec3 lightPos;       // Light position (for point/spot lights)
uniform vec3 lightDir;       // Light direction (for directional/spot lights)
uniform vec3 ambient;        // Ambient light color
uniform vec3 diffuse;        // Diffuse light color
uniform vec3 specular;       // Specular light color
uniform float cutoff;        // Cutoff angle for spotlights
uniform float exponent;      // Specular exponent
uniform int directional;     // 1 for directional light, 0 for point/spotlight
uniform vec3 attenuation;    // Attenuation factors (constant, linear, quadratic)
uniform vec3 viewPos;
uniform vec3 headlightDir;

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

void main()
{   

    float leftBound = (left + 1)/2 *(n_env_frag*res_frag);
    float rightBound = (right + 1)/2 *(n_env_frag*res_frag);
    
    // Check if the fragment is within the bounds
    if (gl_FragCoord.x < leftBound || gl_FragCoord.x > rightBound)
    {
        discard; // Discard the fragment if it's outside the bounds
    }
    
    vec3 norm = normalize(Normal);
    vec3 result = ambient * vColor.rgb; 

    if (directional == 0)
    {	
        // Point/Spot light calculations
        vec3 lightDirection = normalize(lightPos - FragPos);

        float distance = length(lightPos - FragPos);
        float attenuationFactor = 1.0 / (attenuation.x + attenuation.y * distance + attenuation.z * (distance * distance));

        // Diffuse shading
        float diff = max(dot(norm, lightDirection), 0.0);
        vec3 diffuseApplied = diff * diffuse * vColor.rgb * attenuationFactor;
        result += diffuseApplied;

        // Specular shading
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDirection, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), exponent);
        vec3 specularApplied = spec * specular * vColor.rgb * attenuationFactor;
        result += specularApplied;


    }
    else
    {
        // Directional light calculations
        vec3 dir = normalize(-lightDir);

        // Diffuse shading
        float diff = max(dot(norm, dir), 0.0);
        vec3 diffuseApplied = diff * diffuse * vColor.rgb;
        result += diffuseApplied;

        // Specular shading
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-dir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), exponent);
        vec3 specularApplied = spec * specular * vColor.rgb;
        result += specularApplied;
    }

    vec3 ambientHeadlight = vec3(0.1);
    vec3 diffuseHeadlight = vec3(0.4);
    vec3 specularHeadlight = vec3(0.5);
    
    result += ambientHeadlight * vColor.rgb; 
    
    vec3 hDir = normalize(-headlightDir);
    
    // Diffuse shading
    float diff = max(dot(norm, hDir), 0.0);
    vec3 hDiffuseApplied = diff * diffuseHeadlight * vColor.rgb;
    result += hDiffuseApplied;

    // Specular shading
    vec3 viewHDir = normalize(viewPos - FragPos);
    vec3 reflectHDir = reflect(-hDir, norm);
    float spec = pow(max(dot(viewHDir, reflectHDir), 0.0), exponent);
    vec3 hSpecularApplied = spec * specularHeadlight * vColor.rgb;
    result += hSpecularApplied;
    
    FragColor = vec4(result, vColor.a);
    
    int tex_id = int(texID);

    if (tex_id == 1) {
        FragColor = texture(tex0, texCoords) * vec4(result, vColor.a);
    } else if (tex_id == 2) {
        FragColor = texture(tex1, texCoords) * vec4(result, vColor.a);
    } else if (tex_id == 3) {
        FragColor = texture(tex2, texCoords) * vec4(result, vColor.a);
    } else {
        FragColor = vec4(result, vColor.a);
    }
    
}
