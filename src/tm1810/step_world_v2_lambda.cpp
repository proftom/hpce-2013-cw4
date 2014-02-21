#include "heat.hpp"

#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>

//Added for windows
#include <string>
#include <algorithm>

using namespace std;


namespace hpce{
	namespace tm1810 {

		void kernal_xy(const uint32_t *world_properties, const float *world_state, float *buffer, uint32_t x, uint32_t y, uint32_t w, float outer)
		{
			unsigned index = y*w + x;
			float inner = 1 - outer / 4;				// Anything that doesn't spread stays

			if ((world_properties[index] & Cell_Fixed) || (world_properties[index] & Cell_Insulator)){
				
				// Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
				buffer[index] = world_state[index];
			}
			else{
				float contrib = inner;
				float acc = inner*world_state[index];

				// Cell above
				if (!(world_properties[index - w] & Cell_Insulator)) {
					contrib += outer;
					acc += outer * world_state[index - w];
				}

				// Cell below
				if (!(world_properties[index + w] & Cell_Insulator)) {
					contrib += outer;
					acc += outer * world_state[index + w];
				}

				// Cell left
				if (!(world_properties[index - 1] & Cell_Insulator)) {
					contrib += outer;
					acc += outer * world_state[index - 1];
				}

				// Cell right
				if (!(world_properties[index + 1] & Cell_Insulator)) {
					contrib += outer;
					acc += outer * world_state[index + 1];
				}

				// Scale the accumulate value by the number of places contributing to it
				float res = acc / contrib;
				// Then clamp to the range [0,1]
				res = std::min(1.0f, std::max(0.0f, res));
				buffer[index] = res;

			}
		}

		//! Reference world stepping program
		/*! \param dt Amount to step the world by.  Note that large steps will be unstable.
			\param n Number of times to step the world
			\note Overall time increment will be n*dt
			*/
		void StepWorldV2Lambda(world_t &world, float dt, unsigned n)
		{
			unsigned w = world.w, h = world.h;

			float outer = world.alpha*dt;		// We spread alpha to other cells per time

			// This is our temporary working space
			std::vector<float> buffer(w*h);

			for (unsigned t = 0; t < n; t++){
				for (unsigned y = 0; y < h; y++){
					for (unsigned x = 0; x < w; x++){
						
						kernal_xy((uint32_t *)&world.properties[0], &world.state[0], &buffer[0], x, y, w, outer);
					}  // end of for(x...
				} // end of for(y...

				// All cells have now been calculated and placed in buffer, so we replace
				// the old state with the new state
				std::swap(world.state, buffer);
				// Swapping rather than assigning is cheaper: just a pointer swap
				// rather than a memcpy, so O(1) rather than O(w*h)

				world.t += dt; // We have moved the world forwards in time

			} // end of for(t...
		}

	};
}; // namepspace hpce


int main(int argc, char *argv[])
{
	float dt = 0.1;
	unsigned n = 1;
	bool binary = false;

	if (argc>1){
		dt = strtof(argv[1], NULL);
	}
	if (argc>2){
		n = atoi(argv[2]);
	}
	if (argc>3){
		if (atoi(argv[3]))
			binary = true;
	}

	try{
		hpce::world_t world = hpce::LoadWorld(std::cin);
		std::cerr << "Loaded world with w=" << world.w << ", h=" << world.h << std::endl;

		std::cerr << "Stepping by dt=" << dt << " for n=" << n << std::endl;
		hpce::tm1810::StepWorldV2Lambda(world, dt, n);


		hpce::SaveWorld(std::cout, world, binary);
	}
	catch (const std::exception &e){
		std::cerr << "Exception : " << e.what() << std::endl;
		return 1;
	}

	return 0;
}