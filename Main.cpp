//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Mikitovics Márk
// Neptun : I3L1O7
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator*(const float fl) {
		vec4 result(v[0] * fl, v[1] * fl, v[2] * fl, 1);
		return result;
	}

	vec4 operator+(const vec4& vec) {
		vec4 result(v[0]+vec.v[0], v[1]+vec.v[1], v[2]+vec.v[2], 1);
		return result;
	}

	vec4 operator+=(const vec4& vec) {
		this->v[0] += vec.v[0];
		this->v[1] += vec.v[1];
		this->v[2] += vec.v[2];
		return *this;
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
public:
	Camera() {
		Animate(0);
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
	float angle;
	float oldx, oldy, newx, newy;
public:
	Triangle() {
		Animate(0);
	}

	void Create(vec4 a, vec4 b, vec4 c, float colors[9]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[] = { 10.0f*a.v[0], 10.0f*a.v[1], 10.0f*b.v[0], 10.0f*b.v[1], 10.0f*c.v[0], 10.0f*c.v[1] };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		float vertexColors[] = { colors[0], colors[1], colors[2],
										colors[3], colors[4], colors[5],
										colors[6], colors[7], colors[8] };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
		angle = 0;
	}

	void AnimateBycicle(float x1, float y1, float x2, float y2, float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0;// 4 * cosf(t / 2);
		wTy = 0;// 4 * sinf(t / 2);
		angle = t;
		oldx = x1;
		oldy = y1;
		newx = x2;
		newy = y2;
	}

	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}

	void DrawBicycle() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 Rotate(cosf(angle), sinf(angle), 0, 0,
			-sinf(angle), cosf(angle), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		mat4 toOrigo(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			-oldx, -oldy, 0, 1);

		mat4 toOrigin(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			newx, newy, 0, 1);

			mat4 MVPTransform = Mscale * Mtranslate * toOrigo * Rotate * toOrigin * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}

};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[100]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		if (nVertices >= 20) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

class LagrangeCurve {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	std::vector<vec4> cps;		// interleaved data of coordinates and colors
	std::vector<float> ts;			// parameter (knot) values
	std::vector<float> res; // all points
	int nRes = 0;

	float L(int i, float t) {
		float Li = 1.0f;
		for (int j = 0; j < cps.size(); j++) {
			if (j != i) {
				Li *= (t - ts[j]) / (ts[i] - ts[j]);
			}
		}
		return Li;
	}
public:
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddControlPoint(float cX, float cY) {

		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		cps.push_back(wVertex);

		float ti = cps.size(); 	// or something better
		ts.push_back(ti);

		// fill interleaved data
		res.clear();
		nRes = 0;

		float length = 0;

		if (cps.size() > 1) {
			for (int i = 1; i < cps.size(); i++) {
				for (int j = 0; j < 100; j++) {
					vec4 tmp = Interpolate(i + 0.01*j);
					res.push_back(tmp.v[0]); //x
					res.push_back(tmp.v[1]); //y
					res.push_back(1); //r
					res.push_back(1); //g
					res.push_back(1); //b
					nRes++;

					if (res.size() > 1) {
						vec4 previous = res[res.size() - 2];
						length += sqrtf(pow(previous.v[0] - tmp.v[0], 2) + pow(previous.v[1] - tmp.v[1], 2));
					}
				}
			}
			length += sqrtf(pow(cps[cps.size() - 2].v[0] - cps[cps.size() - 1].v[0], 2) + pow(cps[cps.size() - 2].v[1] - cps[cps.size() - 1].v[1], 2));
		}
		res.push_back(cps[cps.size()-1].v[0]); //x
		res.push_back(cps[cps.size()-1].v[1]); //y
		res.push_back(1); //r
		res.push_back(1); //g
		res.push_back(1); //b
		nRes++;
		// copy data to the GPU
		printf("Curve length: %f m.\n", length);
		glBufferData(GL_ARRAY_BUFFER, nRes * 5 * sizeof(float), &res[0], GL_DYNAMIC_DRAW);
	}

	vec4 Interpolate(float t) {
		vec4 rr(0, 0, 0, 1);
		for (int i = 0; i < cps.size(); i++) {
			rr += cps[i] * L(i, t);
		}
		return rr;	
	}
	void Draw() {
		if (nRes > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nRes);
		}
	};

	std::vector<float> getPoints() {
		return res;
	}
};

class BezierSurface {
	float heights[25] = {
		0.1f, 0.1f, 0.5f, 1.0f, 1.0f,
		0.5f, 0.5f, 0.5f, 0.5f, 1.0f,
		0.5f, 1.0f, 1.0f, 0.5f, 0.5f,
		0.5f, 1.0f, 1.0f, 0.5f, 0.1f,
		0.5f, 0.5f, 0.5f, 0.1f, 0.1f,
	};
	std::vector<vec4> surfacePoints;
	std::vector<vec4> cps;
	std::vector<Triangle> surfaceTriangles;
	int size = 5;

	int factorial(int n)
	{
		if (n > 1) {
			return n * factorial(n - 1);
		}
		else {
			return 1;
		}
	}

	int kCombination(int n, int k) {
		return factorial(n) / (factorial(k) * (factorial(n - k)));
	}

	float B(int n, int i, float u) {
		return (float)kCombination(n, i) * pow(u, i) * pow(1 - u, n - i);
	}

	float weight(float u, float v) {
		float r = 0;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				r += cps[i*size + j].v[2] * B(size - 1, i, u) * B(size - 1, j, v);
			}
		}
		return r;
	}
	float convertHeightToGreen(float height) {
		return (1.0f - height);
	}

public:
	void Create() {
		// adding control points
		float step = 0.5f;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++)
			{
				vec4 cp = vec4((-1 + j*step), (1 - i*step), heights[i*size + j], 1);
				cps.push_back(cp);
			}
		}
		// calculating points every 50m
		float distance = 0.1f;
		for (int i = 0; i < 21; i++) {
			for (int j = 0; j < 21; j++) {
				float height = weight((float)i / 20.0f, (float)j / 20.0f);
				surfacePoints.push_back(vec4((-1 + j*distance), (1 - i*distance), height, 1));
			}
		}
		// 20 because we do not want to overindex
		for (int i = 0; i < 20; i++) {
			for (int j = 0; j < 20; j++) {

				float r = 0.55f;
				float b = 0.1f;

				Triangle topLeft, bottomRight;
				float colors1[9] = {r, convertHeightToGreen(surfacePoints[i*21 + j].v[2]), b,
									r, convertHeightToGreen(surfacePoints[i*21+j+1].v[2]), b,
									r, convertHeightToGreen(surfacePoints[(i+1)*21+j].v[2]), b };
				topLeft.Create(surfacePoints[i*21+j], surfacePoints[i * 21 + j + 1], surfacePoints[(i + 1) * 21 + j], colors1);

				float colors2[9] = { r, convertHeightToGreen(surfacePoints[i * 21 + j + 1].v[2]), b,
					r, convertHeightToGreen(surfacePoints[(i + 1) * 21 + j].v[2]), b,
					r, convertHeightToGreen(surfacePoints[(i + 1) * 21 + j + 1].v[2]), b };
				bottomRight.Create(surfacePoints[i * 21 + j + 1], surfacePoints[(i + 1) * 21 + j], surfacePoints[(i + 1) * 21 + j + 1], colors2);

				surfaceTriangles.push_back(topLeft);
				surfaceTriangles.push_back(bottomRight);
			}
		}

	}

	std::vector<vec4> GetSurfacePoints() {
		return surfacePoints;
	}

	void Draw() {
		for (int i = 0; i < surfaceTriangles.size(); i++) {
			surfaceTriangles[i].Draw();
		}
	}

};

class Bicycle {
	std::vector<Triangle> parts;
	std::vector<long> timeStamps;
	long startTime;
	LagrangeCurve lagrangeCurve;
	BezierSurface bezierSurface;
	boolean started = false;
	float colors[9] = { 0, 0, 1.0f, 0, 0, 0, 1.0f, 1.0f, 1.0f };


public:

	void Create(LagrangeCurve& lc, BezierSurface& bs) {
		lagrangeCurve = lc;
		bezierSurface = bs;
	}

	void Start() {
		if (!started) {
			started = true;
			startTime = glutGet(GLUT_ELAPSED_TIME);
			for (int i = timeStamps.size() - 1; i > -1 ; i--) {
				timeStamps[i] -= timeStamps[0];
			}
			Create();
		}
	}

	void Create() {
		Triangle left, right;
		std::vector<float> curvePoints;
		curvePoints = lagrangeCurve.getPoints();
		vec4 front, leftWing, rightWing, back;

		float x = curvePoints[0] / 10.0f;
		float y = curvePoints[1] / 10.0f;

		front = vec4(x, y, 0, 1);
		leftWing = vec4(x - 0.04f, y - 0.07f, 0, 1);
		rightWing = vec4(x + 0.04f, y - 0.07f, 0, 1);
		back = vec4(x, y - 0.04f, 0, 1);

		left.Create(front, leftWing, back, colors);
		right.Create(front, rightWing, back, colors);

		parts.push_back(left);
		parts.push_back(right);
	}

	void AddTime(long time) {
		timeStamps.push_back(time);
	}

	int GetIndex(long currentTime) {
		long elapsedTime = currentTime - startTime;
		long step = 0;
		long timePart = 0;
		int skip = 0;
		if (elapsedTime > timeStamps[timeStamps.size() - 1]) {
			elapsedTime = elapsedTime % timeStamps[timeStamps.size() - 1];
		}
		for (int i = 0; i < timeStamps.size() - 1; i++) {
			if ((elapsedTime > timeStamps[i]) && (elapsedTime < timeStamps[i+1])) {
				timePart = elapsedTime - timeStamps[i] ;
				step = (timeStamps[i + 1] - timeStamps[i]) / 100.0f;
				skip = i;
				break;
			}
		}
		return step == 0 ? step : skip*100 + round(timePart / step);
	}

	float CalculateAngle(float x1, float y1, float x2, float y2) {
		float ux = 0;
		float uy = 1.0f;

		float vx = x2 - x1;
		float vy = y2 - y1;

		float ulength = 1.0f;
		float vlength = sqrtf(pow(vx, 2) + pow(vy, 2));

		float dot = ux * vx + uy * vy;

		float cosfi = dot / (ulength * vlength);

		return vx > 0 ? -acosf(cosfi) : acosf(cosfi);
	}

	float GetHeight(float x, float y) {
		std::vector<vec4> surfacePoints = bezierSurface.GetSurfacePoints();
		return x;
	}

	void Animate(float t) {
		if (parts.size() > 0) {
			std::vector<float> curvePoints;
			curvePoints = lagrangeCurve.getPoints();

			long currentTime = glutGet(GLUT_ELAPSED_TIME);

			int index = GetIndex(currentTime);

			if (index >= curvePoints.size() / 5) {
				index = 1;
				float oldx = curvePoints[0];
				float oldy = curvePoints[1];

				float angle = CalculateAngle(curvePoints[0], curvePoints[1], curvePoints[0 + 5], curvePoints[1 + 5]);

				parts[0].AnimateBycicle(oldx, oldy, oldx, oldy, angle);
				parts[1].AnimateBycicle(oldx, oldy, oldx, oldy, angle);
			}

			if (index > 1 && index < (curvePoints.size() / 5) - 1) {
				float oldx = curvePoints[0];
				float oldy = curvePoints[1];

				float newx = curvePoints[index * 5];
				float newy = curvePoints[index * 5 + 1];

				float angle = CalculateAngle(newx, newy, curvePoints[(index + 1) * 5], curvePoints[(index + 1) * 5 + 1]);

				float kek = GetHeight(newx, newy);

				parts[0].AnimateBycicle(oldx, oldy, newx, newy, angle);
				parts[1].AnimateBycicle(oldx, oldy, newx, newy, angle);
			} 
		}
	}

	void Draw() {
		if (started) {
			parts[0].DrawBicycle();
			parts[1].DrawBicycle();
		}
	}
};

// The virtual world: collection of two objects
BezierSurface bezierSurface;
LagrangeCurve lagrangeCurve;
Bicycle bicycle;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	lagrangeCurve.Create();
	bezierSurface.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	bezierSurface.Draw();
	lagrangeCurve.Draw();
	bicycle.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	}
	if (key == ' ') {
		bicycle.Create(lagrangeCurve, bezierSurface);
		bicycle.Start();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lagrangeCurve.AddControlPoint(cX, cY);
		long time = glutGet(GLUT_ELAPSED_TIME);
		bicycle.AddTime(time);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	bicycle.Animate(sec);					// animate the bicycle object
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
