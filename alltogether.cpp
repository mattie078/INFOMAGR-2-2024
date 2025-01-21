#include "precomp.h"
#include "alltogether.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 6: all together now.
// This version shows how to build and maintain a BVH using
// a TLAS (top level acceleration structure) over a collection of
// BLAS'es (bottom level accstructs), with instancing.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new AllTogetherApp(); }

// functions

void IntersectTri( Ray& ray, const Tri& tri )
{
	const float3 edge1 = tri.vertex1 - tri.vertex0;
	const float3 edge2 = tri.vertex2 - tri.vertex0;
	const float3 h = cross( ray.D, edge2 );
	const float a = dot( edge1, h );
	if (a > -0.00001f && a < 0.00001f) return; // ray parallel to triangle
	const float f = 1 / a;
	const float3 s = ray.O - tri.vertex0;
	const float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	const float3 q = cross( s, edge1 );
	const float v = f * dot( ray.D, q );
	if (v < 0 || u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0.0001f) ray.t = min( ray.t, t );
}

inline float IntersectAABB( const Ray& ray, const float3 bmin, const float3 bmax )
{
	float tx1 = (bmin.x - ray.O.x) * ray.rD.x, tx2 = (bmax.x - ray.O.x) * ray.rD.x;
	float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
	float ty1 = (bmin.y - ray.O.y) * ray.rD.y, ty2 = (bmax.y - ray.O.y) * ray.rD.y;
	tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
	float tz1 = (bmin.z - ray.O.z) * ray.rD.z, tz2 = (bmax.z - ray.O.z) * ray.rD.z;
	tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

float IntersectAABB_SSE( const Ray& ray, const __m128& bmin4, const __m128& bmax4 )
{
	static __m128 mask4 = _mm_cmpeq_ps( _mm_setzero_ps(), _mm_set_ps( 1, 0, 0, 0 ) );
	__m128 t1 = _mm_mul_ps( _mm_sub_ps( _mm_and_ps( bmin4, mask4 ), ray.O4 ), ray.rD4 );
	__m128 t2 = _mm_mul_ps( _mm_sub_ps( _mm_and_ps( bmax4, mask4 ), ray.O4 ), ray.rD4 );
	__m128 vmax4 = _mm_max_ps( t1, t2 ), vmin4 = _mm_min_ps( t1, t2 );
	float tmax = min( vmax4.m128_f32[0], min( vmax4.m128_f32[1], vmax4.m128_f32[2] ) );
	float tmin = max( vmin4.m128_f32[0], max( vmin4.m128_f32[1], vmin4.m128_f32[2] ) );
	if (tmax >= tmin && tmin < ray.t && tmax > 0) return tmin; else return 1e30f;
}

// BVH class implementation

BVH::BVH( char* triFile, int N )
{
	FILE* file = fopen( triFile, "r" );
	triCount = N;
	tri = new Tri[N];
	for (int t = 0; t < N; t++) fscanf( file, "%f %f %f %f %f %f %f %f %f\n",
		&tri[t].vertex0.x, &tri[t].vertex0.y, &tri[t].vertex0.z,
		&tri[t].vertex1.x, &tri[t].vertex1.y, &tri[t].vertex1.z,
		&tri[t].vertex2.x, &tri[t].vertex2.y, &tri[t].vertex2.z );
	bvhNode = (BVHNode*)_aligned_malloc( sizeof( BVHNode ) * N * 2, 64 );
	triIdx = new uint[N];
	Build();
}

void BVH::Intersect( Ray& ray )
{
	BVHNode* node = &bvhNode[0], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			for (uint i = 0; i < node->triCount; i++)
				IntersectTri( ray, tri[triIdx[node->leftFirst + i]] );
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		BVHNode* child1 = &bvhNode[node->leftFirst];
		BVHNode* child2 = &bvhNode[node->leftFirst + 1];
	#ifdef USE_SSE
		float dist1 = IntersectAABB_SSE( ray, child1->aabbMin4, child1->aabbMax4 );
		float dist2 = IntersectAABB_SSE( ray, child2->aabbMin4, child2->aabbMax4 );
	#else
		float dist1 = IntersectAABB( ray, child1->aabbMin, child1->aabbMax );
		float dist2 = IntersectAABB( ray, child2->aabbMin, child2->aabbMax );
	#endif
		if (dist1 > dist2) { swap( dist1, dist2 ); swap( child1, child2 ); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}

void BVH::Refit()
{
	Timer t;
	for (int i = nodesUsed - 1; i >= 0; i--) if (i != 1)
	{
		BVHNode& node = bvhNode[i];
		if (node.isLeaf())
		{
			// leaf node: adjust bounds to contained triangles
			UpdateNodeBounds( i );
			continue;
		}
		// interior node: adjust bounds to child node bounds
		BVHNode& leftChild = bvhNode[node.leftFirst];
		BVHNode& rightChild = bvhNode[node.leftFirst + 1];
		node.aabbMin = fminf( leftChild.aabbMin, rightChild.aabbMin );
		node.aabbMax = fmaxf( leftChild.aabbMax, rightChild.aabbMax );
	}
	printf( "BVH refitted in %.2fms  ", t.elapsed() * 1000 );
}

void BVH::Build()
{
	// reset node pool
	nodesUsed = 2;
	// populate triangle index array
	for (uint i = 0; i < triCount; i++) triIdx[i] = i;
	// calculate triangle centroids for partitioning
	for (uint i = 0; i < triCount; i++)
		tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
	// assign all triangles to root node
	BVHNode& root = bvhNode[0];
	root.leftFirst = 0, root.triCount = triCount;
	UpdateNodeBounds( 0 );
	// subdivide recursively
	Timer t;
	Subdivide( 0 );
	printf( "BVH constructed in %.2fms  ", t.elapsed() * 1000 );
}

void BVH::UpdateNodeBounds( uint nodeIdx )
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3( 1e30f );
	node.aabbMax = float3( -1e30f );
	for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
	{
		uint leafTriIdx = triIdx[first + i];
		Tri& leafTri = tri[leafTriIdx];
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex0 );
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex1 );
		node.aabbMin = fminf( node.aabbMin, leafTri.vertex2 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex0 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex1 );
		node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex2 );
	}
}

float BVH::FindBestSplitPlane( BVHNode& node, int& axis, float& splitPos )
{
	float bestCost = 1e30f;
	for (int a = 0; a < 3; a++)
	{
		float boundsMin = 1e30f, boundsMax = -1e30f;
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			boundsMin = min( boundsMin, triangle.centroid[a] );
			boundsMax = max( boundsMax, triangle.centroid[a] );
		}
		if (boundsMin == boundsMax) continue;
		// populate the bins
		struct Bin { aabb bounds; int triCount = 0; } bin[BINS];
		float scale = BINS / (boundsMax - boundsMin);
		for (uint i = 0; i < node.triCount; i++)
		{
			Tri& triangle = tri[triIdx[node.leftFirst + i]];
			int binIdx = min( BINS - 1, (int)((triangle.centroid[a] - boundsMin) * scale) );
			bin[binIdx].triCount++;
			bin[binIdx].bounds.grow( triangle.vertex0 );
			bin[binIdx].bounds.grow( triangle.vertex1 );
			bin[binIdx].bounds.grow( triangle.vertex2 );
		}
		// gather data for the 7 planes between the 8 bins
		float leftArea[BINS - 1], rightArea[BINS - 1];
		int leftCount[BINS - 1], rightCount[BINS - 1];
		aabb leftBox, rightBox;
		int leftSum = 0, rightSum = 0;
		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bin[i].triCount;
			leftCount[i] = leftSum;
			leftBox.grow( bin[i].bounds );
			leftArea[i] = leftBox.area();
			rightSum += bin[BINS - 1 - i].triCount;
			rightCount[BINS - 2 - i] = rightSum;
			rightBox.grow( bin[BINS - 1 - i].bounds );
			rightArea[BINS - 2 - i] = rightBox.area();
		}
		// calculate SAH cost for the 7 planes
		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
			if (planeCost < bestCost)
				axis = a, splitPos = boundsMin + scale * (i + 1), bestCost = planeCost;
		}
	}
	return bestCost;
}

void BVH::Subdivide( uint nodeIdx )
{
	// terminate recursion
	BVHNode& node = bvhNode[nodeIdx];
	// determine split axis using SAH
	int axis;
	float splitPos;
	float splitCost = FindBestSplitPlane( node, axis, splitPos );
	float nosplitCost = node.CalculateNodeCost();
	if (splitCost >= nosplitCost) return;
	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (tri[triIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap( triIdx[i], triIdx[j--] );
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;
	// create child nodes
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds( leftChildIdx );
	UpdateNodeBounds( rightChildIdx );
	// recurse
	Subdivide( leftChildIdx );
	Subdivide( rightChildIdx );
}

// BVHInstance implementation

void BVHInstance::SetTransform( mat4& transform )
{
	invTransform = transform.Inverted();
	// calculate world-space bounds using the new matrix
	float3 bmin = bvh->bvhNode[0].aabbMin, bmax = bvh->bvhNode[0].aabbMax;
	bounds = aabb();
	for (int i = 0; i < 8; i++)
		bounds.grow( TransformPosition( float3( i & 1 ? bmax.x : bmin.x,
			i & 2 ? bmax.y : bmin.y, i & 4 ? bmax.z : bmin.z ), transform ) );
}

void BVHInstance::Intersect( Ray& ray )
{
	// backup ray and transform original
	Ray backupRay = ray;
	ray.O = TransformPosition( ray.O, invTransform );
	ray.D = TransformVector( ray.D, invTransform );
	ray.rD = float3( 1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z );
	// trace ray through BVH
	bvh->Intersect( ray );
	// restore ray origin and direction
	backupRay.t = ray.t;
	ray = backupRay;
}

// TLAS implementation

TLAS::TLAS( BVHInstance* bvhList, int N )
{
	// copy a pointer to the array of bottom level accstruc instances
	blas = bvhList;
	blasCount = N;
	// allocate TLAS nodes
	tlasNode = (TLASNode*)_aligned_malloc( sizeof( TLASNode ) * 2 * N, 64 );
	nodesUsed = 2;
}

int TLAS::FindBestMatch( int* list, int N, int A )
{
	// find BLAS B that, when joined with A, forms the smallest AABB
	float smallest = 1e30f;
	int bestB = -1;
	for (int B = 0; B < N; B++) if (B != A)
	{
		float3 bmax = fmaxf( tlasNode[list[A]].aabbMax, tlasNode[list[B]].aabbMax );
		float3 bmin = fminf( tlasNode[list[A]].aabbMin, tlasNode[list[B]].aabbMin );
		float3 e = bmax - bmin;
		float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
		if (surfaceArea < smallest) smallest = surfaceArea, bestB = B;
	}
	return bestB;
}

void TLAS::Build()
{
	// assign a TLASleaf node to each BLAS
	int nodeIdx[INSTANCE_AMOUNT], nodeIndices = blasCount;
	nodesUsed = 1;
	for (uint i = 0; i < blasCount; i++)
	{
		nodeIdx[i] = nodesUsed;
		tlasNode[nodesUsed].aabbMin = blas[i].bounds.bmin;
		tlasNode[nodesUsed].aabbMax = blas[i].bounds.bmax;
		tlasNode[nodesUsed].BLAS = i;
		tlasNode[nodesUsed++].leftRight = 0; // makes it a leaf
	}
	// use agglomerative clustering to build the TLAS
	int A = 0, B = FindBestMatch( nodeIdx, nodeIndices, A );
	while (nodeIndices > 1)
	{
		int C = FindBestMatch( nodeIdx, nodeIndices, B );
		if (A == C)
		{
			int nodeIdxA = nodeIdx[A], nodeIdxB = nodeIdx[B];
			TLASNode& nodeA = tlasNode[nodeIdxA];
			TLASNode& nodeB = tlasNode[nodeIdxB];
			TLASNode& newNode = tlasNode[nodesUsed];
			newNode.leftRight = nodeIdxA + (nodeIdxB << 16);
			newNode.aabbMin = fminf( nodeA.aabbMin, nodeB.aabbMin );
			newNode.aabbMax = fmaxf( nodeA.aabbMax, nodeB.aabbMax );
			nodeIdx[A] = nodesUsed++;
			nodeIdx[B] = nodeIdx[nodeIndices - 1];
			B = FindBestMatch( nodeIdx, --nodeIndices, A );
		}
		else A = B, B = C;
	}
	tlasNode[0] = tlasNode[nodeIdx[A]];
}

void TLAS::BuildWithBraiding()
{
    // 1) Build an array of BRefs, each representing the root node
    //    of one BLAS (instance).
    std::vector<BRef> brefs;
    brefs.reserve(blasCount);

    for (int i = 0; i < blasCount; i++)
    {
        BRef bref;
        bref.ref       = &blas[i].bvh->bvhNode[0]; // the BLAS root
        bref.bounds    = blas[i].bounds;           // the instance's world-space bounding box
        bref.objectID  = i;
        bref.numPrims  = blas[i].bvh->triCount;    // total number of triangles in that BLAS
        brefs.push_back(bref);
    }

    // Reset counters, if you rely on them, e.g.:
    nodesUsed = 1; // we will store the root in tlasNode[0].

    // 2) Recursively build a top-level BVH with partial opening:
    //    build the entire TLAS in tlasNode[0].
    //    We'll have the recursion return the index of the node it builds.
    BuildRecursive(brefs, 0, (int)brefs.size(), /*destinationNodeIndex=*/0);
}

inline int MaxDimension(const float3& v)
	{
		int maxDim = 0;
		if (v.y > v.x) maxDim = 1;
		if (v.z > (maxDim == 0 ? v.x : v.y)) maxDim = 2;
		return maxDim;
	}


// Recursively builds a TLAS node for BRefs in [start..end),
// storing that node in tlasNode[nodeIndex].
//
// Returns 'nodeIndex' (for convenience), or possibly you could
// return the index of the node just created.
int TLAS::BuildRecursive(std::vector<BRef>& brefs, int start, int end, int nodeIndex)
{
    // 1) Base case / termination checks
    const int count = end - start;
    if (count <= 0) return nodeIndex; // no references => degenerate
    if (count == 1)
    {
        // We have exactly one BRef => this is a leaf in the TLAS.
        TLASNode& leaf = tlasNode[nodeIndex];
        leaf.aabbMin = brefs[start].bounds.bmin;
        leaf.aabbMax = brefs[start].bounds.bmax;
        // We store the BLAS ID in 'leaf.BLAS' and mark 'leftRight=0' => leaf
        leaf.BLAS = brefs[start].objectID;
        leaf.leftRight = 0;  // signals leaf
        return nodeIndex;
    }

    // 2) Compute bounding box of the entire segment => store in this node
    float3 segBMin(1e30f), segBMax(-1e30f);
    for (int i = start; i < end; i++)
    {
        segBMin = fminf(segBMin, brefs[i].bounds.bmin);
        segBMax = fmaxf(segBMax, brefs[i].bounds.bmax);
    }

    // Optional: check if all references have the same objectID,
    // or if there's minimal overlap, etc., as described in your paper.
    // If so, you can skip opening entirely, or just build a leaf, etc.

    // 3) Opening pass: Decide which BRefs to open in this segment.
    //    We'll do a simple approach: gather them, then open them all at once.

    std::vector<int> openList;
    const float3 segExtent = segBMax - segBMin;
    const int    splitDim  = MaxDimension(segExtent);     // 0=x,1=y,2=z
	const float  threshold = 0.1f * (splitDim == 0 ? segExtent.x : (splitDim == 1 ? segExtent.y : segExtent.z));   // 10% threshold

    for (int i = start; i < end; i++)
    {
        if (ShouldOpenNode(brefs[i], splitDim, threshold))
        {
            openList.push_back(i);
        }
    }

    // 4) Perform the "open" by replacing those references with references
    //    to their children in the same array.
    //    NOTE: to avoid messing up the iteration while we add new BRefs,
    //    we often open *after* we collect all candidates. We do it here:
    for (int idx : openList)
    {
        OpenNode(brefs, idx);
    }

    // 5) Now that we've opened whatever we wanted, partition this segment.
    //    That means we find a best axis/plane via SAH binning, then rearrange
    //    [start..end) so that references < plane go left, >= plane go right.
    //    We'll get a 'mid' index out of it.
    int mid = PartitionBRefs(brefs, start, end);

    // 6) Store the bounding box in tlasNode[nodeIndex].
    TLASNode& node = tlasNode[nodeIndex];
    node.aabbMin = segBMin;
    node.aabbMax = segBMax;

    // 7) Create two child indices in the TLAS array. We'll store them in the
    //    16-bit halves of 'leftRight' as in your code. Then we recursively build them.
    int leftChildIndex  = nodesUsed++;
    int rightChildIndex = nodesUsed++;
    node.leftRight = (leftChildIndex & 0xffff) | (rightChildIndex << 16);

    // Recurse:
    BuildRecursive(brefs, start, mid, leftChildIndex);
    BuildRecursive(brefs, mid,   end, rightChildIndex);

    // Return index of this node (optionally).
    return nodeIndex;
}

bool TLAS::ShouldOpenNode(const BRef& bref, int splitDim, float threshold) const
{
	if (bref.ref == nullptr) return false; // invalid reference

    // If already a leaf in the object BVH, can't open
    if (bref.ref->isLeaf()) return false;

    // Check bounding box extent in 'splitDim'
	const float boxExtent = (splitDim == 0) ? (bref.bounds.bmax.x - bref.bounds.bmin.x) :
						   (splitDim == 1) ? (bref.bounds.bmax.y - bref.bounds.bmin.y) :
											 (bref.bounds.bmax.z - bref.bounds.bmin.z);

    // If the subtree's bounding box in that dimension is big,
    // consider opening it:
    return (boxExtent > threshold);
}


void TLAS::OpenNode(std::vector<BRef>& brefs, int idx)
{
    // The BRef we’re about to open:
    BRef parentCopy = brefs[idx];
    BVHNode* node   = parentCopy.ref;

    // If for some reason it's a leaf or invalid, do nothing:
    if (node->isLeaf()) return;

    // Each child is node->leftFirst and node->leftFirst+1 in your BVH.
	BVHNode* leftChild  = &node[node->leftFirst];
	BVHNode* rightChild = &node[node->leftFirst + 1];

    // We create new BRefs for these children:
    BRef leftBRef, rightBRef;

    leftBRef.ref      = leftChild;
    leftBRef.bounds   = ComputeChildWorldBounds(leftChild, parentCopy);
    leftBRef.objectID = parentCopy.objectID;
    leftBRef.numPrims = CountSubtreePrims(leftChild);  // <-- here

    rightBRef.ref      = rightChild;
    rightBRef.bounds   = ComputeChildWorldBounds(rightChild, parentCopy);
    rightBRef.objectID = parentCopy.objectID;
    rightBRef.numPrims = CountSubtreePrims(rightChild); // <-- here

    // Insert them into brefs.
    // Often you'd do a "push_back" for both children:
    brefs.push_back(leftBRef);
    brefs.push_back(rightBRef);

    // Option A: Mark the original BRef as invalid, or remove it from the array:
    //    - Removing it right now changes array indexing if we do it in the middle 
    //      of a for(...) loop. It's often simpler to do it *after* the loop.
    //    - If you do remove it, you might do swap/pop:
    //         std::swap(brefs[idx], brefs.back());
    //         brefs.pop_back();
    //      But then you have to handle iteration carefully.
    //
    // For simplicity, let's just set numPrims=0 to ignore it, or store a sentinel:
    // parentRef.numPrims = 0;
    // parentRef.bounds   = aabb(); // or some invalid bounds
    // parentRef.ref      = nullptr;
}

// In your TLAS class (or in a suitable place where BVHNode is visible):
unsigned int TLAS::CountSubtreePrims(const BVHNode* node) 
{

    if (node == nullptr) return 0;

    // if node is leaf, return its triCount
    if (node->isLeaf()) return node->triCount;

    // -- HACK: check that leftFirst+1 is still within the array
    //    we allocated for the BLAS (N * 2). If out of range, 
    //    just return this node's triCount or 0 to avoid a crash.
    if (node->leftFirst + 1 >= MODEL_TRI_COUNT)
    {
        // fallback: treat it like a leaf or skip it entirely
        // so we don't dereference invalid pointers
        return node->triCount;
        // or 'return 0;' if you prefer
    }

    // If it's in range, proceed as normal
    const BVHNode* leftChild  = &node[node->leftFirst];
    const BVHNode* rightChild = &node[node->leftFirst + 1];
    return CountSubtreePrims(leftChild) + CountSubtreePrims(rightChild);
}



aabb TLAS::ComputeChildWorldBounds(const BVHNode* child, const BRef& parentRef)
{
    // If the child’s bounding box is *already* in world space:
	aabb bounds;
	bounds.grow(child->aabbMin);
	bounds.grow(child->aabbMax);
	return bounds;
}


int TLAS::PartitionBRefs(std::vector<BRef>& brefs, int start, int end)
{
    if (end - start <= 1) return (start + end) >> 1; // trivial

    // 1) Compute centroid bounding box
    float3 cbmin(1e30f), cbmax(-1e30f);
    for (int i = start; i < end; i++)
    {
        float3 c = 0.5f * (brefs[i].bounds.bmin + brefs[i].bounds.bmax);
        cbmin = fminf(cbmin, c);
        cbmax = fmaxf(cbmax, c);
    }
    float3 diag = cbmax - cbmin;
    if (diag.x < 1e-6f && diag.y < 1e-6f && diag.z < 1e-6f)
    {
        // all centroids are basically the same => just do a mid-split
        return (start + end) >> 1;
    }
    // pick axis with greatest extent
    int axis = 0;
    if (diag.y > diag.x) axis = 1;
    if (diag.z > diag[axis]) axis = 2;

    // 2) binning
    float minC = cbmin[axis], maxC = cbmax[axis];
    float extent = maxC - minC;
    if (extent < 1e-6f) return (start + end) >> 1;

    struct Bin
    {
        aabb   bounds;
        int    count   = 0;
        int    prims   = 0; // sum of numPrims for BRefs
    } bin[BINS];

    float scale = (float)BINS / extent;
    for (int i = start; i < end; i++)
    {
        float3 c = 0.5f * (brefs[i].bounds.bmin + brefs[i].bounds.bmax);
        int b = (int)((c[axis] - minC) * scale);
        if (b < 0) b = 0; 
        if (b >= BINS) b = BINS - 1;
        bin[b].count++;
        bin[b].prims += brefs[i].numPrims;
        bin[b].bounds.grow(brefs[i].bounds.bmin);
        bin[b].bounds.grow(brefs[i].bounds.bmax);
    }

    // prefix sums from left
    float  leftArea[BINS], rightArea[BINS];
    int    leftCount[BINS], rightCount[BINS];
    int    leftPrims[BINS], rightPrims[BINS];

    aabb current;
    int   countAccum = 0, primsAccum = 0;
    for (int b = 0; b < BINS; b++)
    {
        current.grow(bin[b].bounds);
        countAccum += bin[b].count;
        primsAccum += bin[b].prims;
        leftArea[b]  = current.area();
        leftCount[b] = countAccum;
        leftPrims[b] = primsAccum;
    }
    // suffix sums from right
    current = aabb();
    countAccum = 0; 
    primsAccum = 0;
    for (int b = BINS - 1; b >= 0; b--)
    {
        current.grow(bin[b].bounds);
        countAccum += bin[b].count;
        primsAccum += bin[b].prims;
        rightArea[b]  = current.area();
        rightCount[b] = countAccum;
        rightPrims[b] = primsAccum;
    }

    // 3) find best split
    float bestCost = 1e30f;
    int bestSplit = -1;
    for (int s = 0; s < BINS - 1; s++)
    {
        // cost = leftArea[s]*leftPrims[s] + rightArea[s+1]*rightPrims[s+1]
        float cost = leftArea[s] * leftPrims[s] + rightArea[s + 1] * rightPrims[s + 1];
        if (cost < bestCost)
        {
            bestCost = cost;
            bestSplit = s;
        }
    }
    if (bestSplit < 0) return (start + end) >> 1;

    // 4) Partition in-place
    float splitPos = minC + (bestSplit + 1) / scale;
    int i = start, j = end - 1;
    while (i <= j)
    {
        float3 c = 0.5f * (brefs[i].bounds.bmin + brefs[i].bounds.bmax);
        if (c[axis] < splitPos) i++;
        else
        {
            std::swap(brefs[i], brefs[j]);
            j--;
        }
    }
    // 'i' is the partition boundary
    // If one side is empty, just do a fallback.
    if (i == start || i == end) return (start + end) >> 1;

    return i;
}



void TLAS::Intersect( Ray& ray )
{
	ray.rD = float3( 1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z );
	TLASNode* node = &tlasNode[0], * stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->isLeaf())
		{
			blas[node->BLAS].Intersect( ray );
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		TLASNode* child1 = &tlasNode[node->leftRight & 0xffff];
		TLASNode* child2 = &tlasNode[node->leftRight >> 16];
		float dist1 = IntersectAABB( ray, child1->aabbMin, child1->aabbMax );
		float dist2 = IntersectAABB( ray, child2->aabbMin, child2->aabbMax );
		if (dist1 > dist2) { swap( dist1, dist2 ); swap( child1, child2 ); }
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) stack[stackPtr++] = child2;
		}
	}
}

// AllTogetherApp implementation

void AllTogetherApp::Init()
{
	BVH* bvh = new BVH( "assets/armadillo.tri", MODEL_TRI_COUNT );
	for (int i = 0; i < INSTANCE_AMOUNT; i++)
		bvhInstance[i] = BVHInstance( bvh );
	tlas = TLAS( bvhInstance, INSTANCE_AMOUNT );

	position = new float3[INSTANCE_AMOUNT];
	direction = new float3[INSTANCE_AMOUNT];
	orientation = new float3[INSTANCE_AMOUNT];
	float scaleFactor = 0.2f; 
	for( int i = 0; i < INSTANCE_AMOUNT; i++ )
	{
		position[i] = float3( RandomFloat(), RandomFloat(), RandomFloat() ) - 0.5f;
		position[i] *= 4;
		direction[i] = normalize( position[i] ) * 0.05f;
		orientation[i] = float3( RandomFloat(), RandomFloat(), RandomFloat() ) * 2.5f;
		bvhInstance[i].SetTransform( mat4::Scale( scaleFactor ) ); 
	}
}

void AllTogetherApp::Tick( float deltaTime )
{
	// animate the scene
	for( int i = 0; i < INSTANCE_AMOUNT; i++ )
	{
		mat4 R = mat4::RotateX( orientation[i].x ) * 
				 mat4::RotateY( orientation[i].y ) *
				 mat4::RotateZ( orientation[i].z ) * mat4::Scale( 0.2f );
		bvhInstance[i].SetTransform( mat4::Translate( position[i] ) * R );
		position[i] += direction[i], orientation[i] += direction[i];
		if (position[i].x < -3 || position[i].x > 3) direction[i].x *= -1;
		if (position[i].y < -3 || position[i].y > 3) direction[i].y *= -1;
		if (position[i].z < -3 || position[i].z > 3) direction[i].z *= -1;
	}
	// update the TLAS
	Timer t;
	// tlas.Build();
	tlas.BuildWithBraiding();
	float tlasTime = t.elapsed() * 1000;
	// draw the scene
	float3 p0( -1, 1, 2 ), p1( 1, 1, 2 ), p2( -1, -1, 2 );
#pragma omp parallel for schedule(dynamic)
	for (int tile = 0; tile < (SCRWIDTH * SCRHEIGHT / 64); tile++)
	{
		int x = tile % (SCRWIDTH / 8), y = tile / (SCRWIDTH / 8);
		Ray ray;
		ray.O = float3( 0, 0, -6.5f );
		for (int v = 0; v < 8; v++) for (int u = 0; u < 8; u++)
		{
			float3 pixelPos = ray.O + p0 +
				(p1 - p0) * ((x * 8 + u) / (float)SCRWIDTH) +
				(p2 - p0) * ((y * 8 + v) / (float)SCRHEIGHT);
			ray.D = normalize( pixelPos - ray.O ), ray.t = 1e30f;
			tlas.Intersect( ray );
			uint c = ray.t < 1e30f ? (int)(255 / (1 + max( 0.f, ray.t - 4 ))) : 0;
			screen->Plot( x * 8 + u, y * 8 + v, c * 0x10101 );
		}
	}
	// report
	float elapsed = t.elapsed() * 1000;
	printf( "tlas build: %.2fms, tracing time: %.2fms (%5.2fK rays/s)\n", tlasTime, elapsed, sqr( 630 ) / elapsed );
}

// EOF