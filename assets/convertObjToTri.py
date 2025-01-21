import sys

def obj_to_tri(obj_file, tri_file):
    vertices = []
    triangles = []

    with open(obj_file, 'r') as obj:
        for line in obj:
            if line.startswith('v '):  # Vertex line
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith('f '):  # Face line
                _, v1, v2, v3 = line.split()
                v0 = vertices[int(v1.split('/')[0]) - 1]
                v1 = vertices[int(v2.split('/')[0]) - 1]
                v2 = vertices[int(v3.split('/')[0]) - 1]
                triangles.append((v0, v1, v2))

    with open(tri_file, 'w') as tri:
        for triangle in triangles:
            v0, v1, v2 = triangle
            tri.write(f"{v0[0]} {v0[1]} {v0[2]} "
                      f"{v1[0]} {v1[1]} {v1[2]} "
                      f"{v2[0]} {v2[1]} {v2[2]}\n")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convertObjToTri.py input.obj output.tri")
    obj_to_tri(sys.argv[1], sys.argv[2])
    