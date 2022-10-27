#version 420

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D tex;
uniform vec3 viewPos;

out vec4 finalColor;

const int resolution_scale = 4;
const int color_depth = 5;

const int pattern[16] = {
    -4,  0, -3,  1,
     2, -2,  3, -1,
    -3,  1, -4,  0,
     3, -1,  2, -2
};

int dithering_pattern(ivec2 fragcoord) {
	int x = fragcoord.x % 4;
	int y = fragcoord.y % 4;

	return pattern[y * 4 + x];
}

void main() {
    ivec2 uv = ivec2(fragTexCoord.xy / float(resolution_scale));
    // gamma correct
    vec3 color = pow(texture(texture0, fragTexCoord).rgb, vec3(1.0/2.2));

    //float zNear = 0.01; // camera z near
    //float zFar = 10.0;  // camera z far
    //float z = texture(texture1, fragTexCoord).x;

    //// Linearize depth value
    //float depth = (2.0*zNear)/(zFar + zNear - z*(zFar - zNear));
    //const float fogDensity = 0.3;
    //const vec3 fogColor = vec3(0.1, 0.1, 0.1);
    //float dist = (1 - depth) * 30;
    //float fogFactor = clamp(1.0/exp((dist*fogDensity)*(dist*fogDensity)), 0.0, 1.0);


    //color = mix(color, fogColor, fogFactor);

    // Convert from [0.0, 1.0] range to [0, 255] range
    ivec3 c = ivec3(round(color * 255.0));

    c += ivec3(dithering_pattern(uv));

    // Truncate from 8 bits to color_depth bits
    c >>= (8 - color_depth);


    // Convert back to [0.0, 1.0] range
    finalColor = vec4(vec3(c) / float(1 << color_depth), 1) * fragColor;
}
