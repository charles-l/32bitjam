#version 420

in vec2 fragTexCoord;
in vec4 fragColor;

uniform sampler2D texture0;
uniform vec4 colDiffuse;

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
	vec3 color = texture(texture0, fragTexCoord).rgb;

	// Convert from [0.0, 1.0] range to [0, 255] range
	ivec3 c = ivec3(round(color * 255.0));

    c += ivec3(dithering_pattern(uv));

	// Truncate from 8 bits to color_depth bits
	c >>= (8 - color_depth);

	// Convert back to [0.0, 1.0] range
	finalColor = vec4(vec3(c) / float(1 << color_depth), 1);
}
