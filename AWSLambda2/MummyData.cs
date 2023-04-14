using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AWSLambda2
{
    public class MummyData
    {
        public float Length { get; set; }
        public float Depth { get; set; }
        public float WestToHead { get; set; }
        public float AgeAtDeath_other { get; set; }
        public float AgeAtDeath_adult { get; set; }
        public float AgeAtDeath_child { get; set; }
        public float AgeAtDeath_infant { get; set; }
        public float WestToFeet { get; set; }
        public float SouthToHead { get; set; }
        public float Wrapping_full { get; set; }
        public float Wrapping_partial { get; set; }
        public float Wrapping_none { get; set; }


        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
            Length, Depth, WestToHead, AgeAtDeath_other, AgeAtDeath_adult, AgeAtDeath_child, AgeAtDeath_infant, WestToFeet, SouthToHead, Wrapping_full, Wrapping_partial, Wrapping_none
            };
            int[] dimensions = new int[] { 1, 12 };
            return new DenseTensor<float>(data, dimensions);
        }
    }
}
