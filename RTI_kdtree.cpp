#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/conversions.h>
#include <thread>
#include <pcl/visualization/pcl_visualizer.h>
#include <algorithm>

using namespace std;
using namespace pcl;


#define DEBUG
#ifdef DEBUG
#define DEBUG_MSG(str) do { cout << str << endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif


int countInter = 0;
int countLeafSize = 0;
PolygonMesh pm;
PointCloud<PointXYZ>::Ptr hand(new PointCloud<PointXYZ>());

PointCloud<PointXYZ>::Ptr intersectionCloud(new PointCloud<PointXYZ>());

PointCloud<PointXYZ>::Ptr extraPoints(new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr missingPoints(new PointCloud<PointXYZ>);

vector <pair<PointXYZ, PointXYZ>> bb;

struct Ray {
	PointXYZ orig;
	Eigen::Vector3f dir;
	Eigen::Vector3f invdir;
	int sign[3];

	Ray(const PointXYZ &orig, const  Eigen::Vector3f &dir) : orig(orig), dir(dir)
	{
		invdir[0] = 1 / dir[0];
		invdir[1] = 1 / dir[1];
		invdir[2] = 1 / dir[2];

		sign[0] = (invdir[0] < 0);
		sign[1] = (invdir[1] < 0);
		sign[2] = (invdir[2] < 0);
	}

};

struct Triangle {
	PointXYZ vertices[3];
	PointXYZ midpt;

	Triangle(PointXYZ& s0, PointXYZ& s1, PointXYZ& s2) {
		vertices[0] = s0;
		vertices[1] = s1;
		vertices[2] = s2;

		midpt = s0;
	}

	bool intersect(const Ray& ray) {		
		++countInter;

		const Eigen::Vector3f p = ray.orig.getVector3fMap();
		const Eigen::Vector3f a = vertices[0].getVector3fMap();
		const Eigen::Vector3f b = vertices[1].getVector3fMap();
		const Eigen::Vector3f c = vertices[2].getVector3fMap();
		const Eigen::Vector3f u = b - a;
		const Eigen::Vector3f v = c - a;

		const Eigen::Vector3f n = u.cross(v);
		const float n_dot_ray = n.dot(ray.dir);

		if (std::fabs(n_dot_ray) < 1e-9)
			return (false);

		const float r = n.dot(a - p) / n_dot_ray;

		if (r < 0)
			return (false);

		const Eigen::Vector3f w = p + r * ray.dir - a;
		const float denominator = u.dot(v) * u.dot(v) - u.dot(u) * v.dot(v);
		const float s_numerator = u.dot(v) * w.dot(v) - v.dot(v) * w.dot(u);
		const float s = s_numerator / denominator;

		if (s < 0 || s > 1)
			return (false);

		const float t_numerator = u.dot(v) * w.dot(u) - u.dot(u) * w.dot(v);
		const float t = t_numerator / denominator;

		if (t < 0 || s + t > 1)
			return (false);

		return (true);

	}

};

struct BoundingBox {
	PointXYZ bounds[2];

	int longestAxis() {
		float x = bounds[1].x - bounds[0].x;
		float y = bounds[1].y - bounds[0].y;
		float z = bounds[1].z - bounds[0].z;

		float longestXy = x > y ? x : y;
		float longestXyz = z > longestXy ? z : longestXy;

		if (longestXyz == x) return 0;
		if (longestXyz == y) return 1;
		if (longestXyz == z) return 2;

	}

	bool intersect(const Ray& r) const
	{
		float t1 = (bounds[0].x - r.orig.x)*r.invdir[0];
		float t2 = (bounds[1].x - r.orig.x)*r.invdir[0];
		float t3 = (bounds[0].y - r.orig.y)*r.invdir[1];
		float t4 = (bounds[1].y - r.orig.y)*r.invdir[1];
		float t5 = (bounds[0].z - r.orig.z)*r.invdir[2];
		float t6 = (bounds[1].z - r.orig.z)*r.invdir[2];

		float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
		float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

		DEBUG_MSG("tmin : " << tmin);
		DEBUG_MSG("tmax : " << tmax);

		if(tmin == tmax) return true; //if tmin == tmax, we are inside the box

		// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
		if (tmax < 0) return false;

		// if tmin > tmax, ray doesn't intersect AABB
		if (tmin > tmax) return false;

		return true;
	}

	void expand(vector<Triangle*> triangles)
	{
		float min_x = FLT_MAX;
		float min_y = FLT_MAX;
		float min_z = FLT_MAX;

		float max_x = -FLT_MAX;
		float max_y = -FLT_MAX;
		float max_z = -FLT_MAX;

		for (const auto& t : triangles) {
			for (int i = 0; i < 3; i++) {
				if (t->vertices[i].x < min_x) min_x = t->vertices[i].x;
				if (t->vertices[i].y < min_y) min_y = t->vertices[i].y;
				if (t->vertices[i].z < min_z) min_z = t->vertices[i].z;
				if (t->vertices[i].x > max_x) max_x = t->vertices[i].x;
				if (t->vertices[i].y > max_y) max_y = t->vertices[i].y;
				if (t->vertices[i].z > max_z) max_z = t->vertices[i].z;
			}
		}

		bounds[0].x = min_x; bounds[0].y = min_y; bounds[0].z = min_z;
		bounds[1].x = max_x; bounds[1].y = max_y; bounds[1].z = max_z;

		bb.push_back(make_pair(bounds[0], bounds[1]));
	}
};

struct KDNode {
	BoundingBox bbox;
	KDNode* left;
	KDNode* right;
	vector<Triangle*> triangles;
	bool isLeaf;

	KDNode() {
		bbox = BoundingBox();
		isLeaf = false;
	}

	void addTriangle(Triangle*& t) {
		triangles.push_back(t);
	}

	/* This function takes last element as pivot, places
	the pivot element at its correct position in sorted
	array, and places all smaller elements to left of 
	pivot and all greater elements to right	of pivot */
	int partition(int axis, int low, int high)
	{
		int i = (low - 1);  // Index of smaller element 

		if (axis == 0) {
			int pivot = triangles[high]->vertices[0].x;    // pivot 

			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than or 
				// equal to pivot 
				if (triangles[j]->vertices[0].x <= pivot)
				{
					i++;    // increment index of smaller element 
					swap(triangles[i], triangles[j]);
				}
			}
			swap(triangles[i + 1], triangles[high]);
		}
		if (axis == 1) {
			int pivot = triangles[high]->vertices[0].y;    // pivot 
	
			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than or 
				// equal to pivot 
				if (triangles[j]->vertices[0].y <= pivot)
				{
					i++;    // increment index of smaller element 
					swap(triangles[i], triangles[j]);
				}
			}
			swap(triangles[i + 1], triangles[high]);
		}
		if (axis == 2) {
			int pivot = triangles[high]->vertices[0].z;    // pivot 

			for (int j = low; j <= high - 1; j++)
			{
				// If current element is smaller than or 
				// equal to pivot 
				if (triangles[j]->vertices[0].z <= pivot)
				{
					i++;    // increment index of smaller element 
					swap(triangles[i], triangles[j]);
				}
			}
			swap(triangles[i + 1], triangles[high]);
		}

		return (i + 1);

	}

	void build(int& count, int depth) {
		count++;

		left = new KDNode();
		right = new KDNode();

		/***************Find triangle midpoint (gravity center)***************/
		PointXYZ midpt = PointXYZ(0, 0, 0);
		for (int i = 0; i < triangles.size(); i++) {
			midpt.x += triangles[i]->midpt.x;
			midpt.y += triangles[i]->midpt.y;
			midpt.z += triangles[i]->midpt.z;

		}

		midpt.x /= triangles.size();
		midpt.y /= triangles.size();
		midpt.z /= triangles.size();

		int axis = bbox.longestAxis();

		/* In the following code, points which coordinate on longestAxis
		is greater than  midpoint will go to the right child node,
		else to the left child node */
		if (axis == 0) {
			for (int i = 0; i < triangles.size(); i++) {
				if (midpt.x > triangles[i]->midpt.x)
					right->addTriangle(triangles[i]);
				else
					left->addTriangle(triangles[i]);
			}
		}
		else if (axis == 1) {

			for (int i = 0; i < triangles.size(); i++) {
				if (midpt.y > triangles[i]->midpt.y)
					right->addTriangle(triangles[i]);
				else
					left->addTriangle(triangles[i]);
			}
		}
		else if (axis == 2) {			
			for (int i = 0; i < triangles.size(); i++) {
				if (midpt.z > triangles[i]->midpt.z)
					right->addTriangle(triangles[i]);
				else 
					left->addTriangle(triangles[i]);
			}
		}

		left->bbox.expand(left->triangles);
		right->bbox.expand(right->triangles);

		//check if there is AT LEAST 1 triangle per child node
		//before calling function again
		if (left->triangles.size() > 1 && right->triangles.size() > 1) {
			left->build(count, ++depth);
			right->build(count, ++depth);
		}
		else {
			isLeaf = true;
			countLeafSize += triangles.size();
		}
	}

	void intersect(const Ray& r, int& count, int depth) {
		DEBUG_MSG(" ");
		DEBUG_MSG("depth : " << depth);
		DEBUG_MSG("number of triangles in node : " << triangles.size());
		if (!(left == NULL)) DEBUG_MSG("number of triangles in left child : " << left->triangles.size());
		if (!(right == NULL)) DEBUG_MSG("number of triangles in right child : " << right->triangles.size());
		
		bool inter = bbox.intersect(r);
		DEBUG_MSG("found an intersection ? " << inter);

		if (inter) {
			if (!isLeaf && left != NULL && left->triangles.size() > 0) {
				DEBUG_MSG("left intersect call");
				left->intersect(r, count, ++depth);
			}
		
			if (!isLeaf && right != NULL && right->triangles.size() > 0) {
				DEBUG_MSG("right intersect call");
				right->intersect(r, count, ++depth);
			}
			// In case we reached a leaf of the tree
			else { 
				for (int i = 0; i < triangles.size(); i++) {
					if (triangles[i]->intersect(r)) {
						count++;
						DEBUG_MSG("found triangle intersection in node of depth " << depth << " with triangle " << triangles[i]->vertices[0] <<
							" " << triangles[i]->vertices[1] << " " << triangles[i]->vertices[2]);
						DEBUG_MSG("bbox : " << bbox.bounds[0] << " " << bbox.bounds[1]);
					}
				}

			}
		}
		else {
			return;
		}
	}
};

void compareResults() {

	PLYReader reader;
	PointCloud<PointXYZ>::Ptr reference(new PointCloud<PointXYZ>);

	reader.read("data/expected_result_2.ply", *reference);

	int count = 0;
	for (int i = 0; i < intersectionCloud->size(); i++) {
		for (int j = 0; j < reference->size(); j++) {
			if (intersectionCloud->points[i].x == reference->points[j].x
				&& intersectionCloud->points[i].y == reference->points[j].y
				&& intersectionCloud->points[i].z == reference->points[j].z) {
				++count;
				break;
			}
			if (j == reference->size() - 1) //current point is not in reference
				extraPoints->push_back(intersectionCloud->points[i]);
		}
	}

	int count2 = 0;
	for (int i = 0; i < reference->size(); i++) {
		for (int j = 0; j < intersectionCloud->size(); j++) {
			if (reference->points[i].x == intersectionCloud->points[j].x
				&& reference->points[i].y == intersectionCloud->points[j].y
				&& reference->points[i].z == intersectionCloud->points[j].z) {
				++count2;
				break;
			}
			if (j == intersectionCloud->size() - 1) //current point is not in result
				missingPoints->push_back(reference->points[i]);
		}
	}
	pcl::io::savePLYFileASCII("extra_points.ply", *extraPoints);

	cout << "*********RESULTS***********" << endl;
	cout << "nb of points expected in intersection : " << reference->size() << endl;
	cout << "nb of points in result : " << intersectionCloud->size() << endl;
	cout << "nb of result cloud points in reference : " << count << endl;
	cout << "nb missing points : " << reference->size() - count << endl;
	cout << "missing points : " << endl;

	for (int i = 0; i < missingPoints->size(); i++) cout << missingPoints->points[i];

	cout << "nb of extra points " << intersectionCloud->size() - count << endl;
	cout << "extra points : " << endl;

	for (int i = 0; i < extraPoints->size(); i++) cout << extraPoints->points[i];

	this_thread::sleep_for(10000ms);
}


int main() {
	PLYReader reader;
	reader.read("data/convex_hull_2.ply", pm);

	OBJReader reader2;
	reader2.read("data/test_hand_downsampled_1000_2.obj", *hand);

	PointCloud<PointXYZ>::Ptr hullCloud(new PointCloud<PointXYZ>());
	fromPCLPointCloud2(pm.cloud, *hullCloud);

	vector <Triangle*> triangles;
	for (int i = 0; i < pm.polygons.size(); i++) {
			PointXYZ s0 = hullCloud->at(pm.polygons[i].vertices[0]);
			PointXYZ s1 = hullCloud->at(pm.polygons[i].vertices[1]);
			PointXYZ s2 = hullCloud->at(pm.polygons[i].vertices[2]);

			Triangle* t(new Triangle(s0, s1, s2));

			triangles.push_back(t);
	}

	KDNode kdn;
	for (int i = 0; i < triangles.size(); i++) {
		kdn.addTriangle(triangles[i]);
	}

	// create the bounding box
	kdn.bbox.expand(triangles);

	clock_t startTime0 = clock();

	int * count;
	int value = 0;
	count = &value;

	// build the kd-tree
	kdn.build(*count, 0);

	DEBUG_MSG("number of nodes : " << *count);

	clock_t testTime0 = clock();
	clock_t timePassed0 = testTime0 - startTime0;
	DEBUG_MSG("time passed for creating the kd tree : " << timePassed0);

	Eigen::Vector3f dir = Eigen::Vector3f(0.264882f, 0.688399f, 0.675237f);

	clock_t startTime = clock();
	
	int i = 0;
	for (int i = 0; i < hand->size(); i++) {
		// First ckeck if hand points are outside the global bouding box of the sphere
		if (!(hand->points[i].x >= kdn.bbox.bounds[0].x
			&& hand->points[i].x <= kdn.bbox.bounds[1].x
			&& hand->points[i].y >= kdn.bbox.bounds[0].y
			&& hand->points[i].y <= kdn.bbox.bounds[1].y
			&& hand->points[i].z >= kdn.bbox.bounds[0].z
			&& hand->points[i].z <= kdn.bbox.bounds[1].z)) continue;

		Ray r(hand->points[i], dir);
		
		int * count;
		int value = 0;
		count = &value;

		// Then go down the kd-tree
		kdn.intersect(r, *count, 0);
		DEBUG_MSG("///////////////////////////////////////////////////////////////////");
		DEBUG_MSG("for hand point " << i << " found " << *count << " intersections");

		if (*count == 1) {
			intersectionCloud->push_back(hand->points[i]);
			DEBUG_MSG("added point " << i << " with coordinates " << hand->points[i].x << " " << hand->points[i].y << " " << hand->points[i].z << " in intersection cloud.");
		}

		++i;
	}

	clock_t testTime = clock();
	clock_t timePassed = testTime - startTime;
	DEBUG_MSG("time passed for calculating intersection : " << timePassed);

	DEBUG_MSG("number of intersections : " << countInter);

	DEBUG_MSG("total number of triangles in leaves : " << countLeafSize);

	compareResults();

	this_thread::sleep_for(10000ms);

	pcl::io::savePLYFile("data/result_intersection_hand_sphere_2_kd-tree_20.ply", *intersectionCloud);

	return 0;
}