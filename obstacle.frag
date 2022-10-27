#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragPosition;

// Input uniform values
uniform vec4 colDiffuse;
uniform int isReflection;

// Output fragment color
out vec4 finalColor;

const float waterlevel = -2.0;

void main()
{
    // NOTE: Implement here your fragment shader code

    if (isReflection == 1 && fragPosition.y < waterlevel) discard;

    finalColor = colDiffuse;
}
