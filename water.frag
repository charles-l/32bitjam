#version 330
precision mediump float;

varying vec2 textureCoords;
varying vec4 clip;

uniform vec4 colDiffuse;
varying vec3 fragPosition;

uniform sampler2D texture0; // reflection

uniform vec3 viewPos;
uniform float time;

void main(void)
{
    vec3 fragToView = normalize(viewPos - fragPosition);

	vec2 normalizedDeviceSpace = (clip.xy/clip.w)/2.0 + 0.5; // fragment coordinates in screen space
    float fresnel = dot(fragToView, vec3(0,1,0));
    fresnel = pow(fresnel, 0.3);

	vec2 reflectTexCoords = vec2(normalizedDeviceSpace.x, 1.0-normalizedDeviceSpace.y);

	reflectTexCoords=clamp(reflectTexCoords, 0.01, 0.99);

    reflectTexCoords.x += cos((reflectTexCoords.y) * 50 + time * 0.5) / clamp(clip.w, 5, 10) / 50;
    reflectTexCoords.y += sin((reflectTexCoords.x) * 60 + time * 0.5) / clamp(clip.w, 5, 10) / 50;

	vec4 reflectColor = texture2D(texture0, reflectTexCoords);

	gl_FragColor = mix(mix(vec4(0, 1, 0, 1), reflectColor, fresnel), colDiffuse, 0.3);
}
