#version 330

varying vec2 textureCoords;
varying vec4 clip;

attribute vec3 vertexPosition;
attribute vec2 vertexTexCoord;

uniform mat4 mvp;
uniform mat4 matModel;

varying vec3 fragPosition;

void main()
{
    fragPosition = vec3(matModel*vec4(vertexPosition, 1.0));
    textureCoords = vertexTexCoord;
    clip = mvp*vec4(vertexPosition, 1.0);
    gl_Position = clip;
}
