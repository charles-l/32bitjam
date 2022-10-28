#version 420
in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;

uniform vec3 viewPos;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform float fogDensity;
uniform vec3 fogColor;

out vec4 finalColor;

void main() {
    float dist = length(viewPos - fragPosition) / 10;
    float fogFactor = clamp(1.0/exp((dist*fogDensity)*(dist*fogDensity)), 0.0, 1.0);

    vec4 texelColor = texture(texture0, fragTexCoord)*colDiffuse*fragColor;
    finalColor = mix(vec4(fogColor, 1), texelColor, fogFactor);
}
