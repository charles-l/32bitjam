#version 330
precision mediump float;

varying vec2 textureCoords;
varying vec4 clip;

uniform vec4 colDiffuse;
varying vec3 fragPosition;

uniform sampler2D texture0; // reflection

uniform vec3 viewPos;

void main(void)
{
    vec3 viewD = normalize(viewPos - fragPosition); // view versor
    //viewD = reflect(viewD, vec3(0, 1, 0));

	vec2 normalizedDeviceSpace = (clip.xy/clip.w)/2.0 + 0.5; // fragment coordinates in screen space

	vec2 reflectTexCoords = vec2(normalizedDeviceSpace.x, 1.0-normalizedDeviceSpace.y);

	reflectTexCoords=clamp(reflectTexCoords, 0.01, 0.99);

	vec4 reflectColor = texture2D(texture0, reflectTexCoords);

	gl_FragColor = mix(reflectColor, colDiffuse, 0.3);
}
