/**
 * Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.sparkprojects

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest

import org.apache.spark.rdd.RDD


/**
 * Provides a way of calculating the two-sided Kolmogorov Smirnov test for data sampled from a
 * continuous distribution. By comparing the largest difference between the empirical cumulative
 * distribution of the sample data and the theoretical distribution (in the case of 1 sample test),
 * or another sample data (in the case of the 2 sample test) we can provide a test for the
 * the null hypothesis that the sample data comes from that theoretical distribution, or that
 * both samples come from the same distribution, in the case of the 1 sample and 2 sample tests,
 * respectively.
 * https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
 */
object KS {
  /**
   * Calculate empirical cumulative distribution function values to then calculate D- and D+
   * @param dat data over which we which to calculate the empirical cumulative distribution
   * @return and RDD of (Double, Double, Double), where the first element in each tuple is the value
   *         and the second element is the value for D- and the third element is the value for D+
   */
  def ecdf(dat: RDD[Double]): RDD[(Double, Double, Double)] = {
    val n = dat.count().toDouble
    dat.sortBy(x => x).zipWithIndex().map { case (v, i) => (v, i / n, (i + 1) / n) }
  }

  /**
   * Runs a KS test for 1 set of sample data, comparing it to a theoretical distribution
   * @param dat the data we wish to evaluate
   * @param cdf a function to calculate the
   * @return the KS statistic and p-value associated with a two sided test
   */
  def testOneSample(dat: RDD[Double], cdf: Double => Double): (Double, Double) = {
    val empiriRDD = ecdf(dat) // empirical distribution
    val distances = empiriRDD.map {
      case (v, dl, dp) => {
        val cdfVal = cdf(v)
        Math.max(cdfVal - dl, dp - cdfVal)
      }
    }
    val ksStat = distances.max()
    evalP(ksStat, distances.count())
  }

  /**
   * Runs a KS test for 1 set of sample data, comparing it to a theoretical distribution. Optimized
   * such that each partition runs a separate mapping operation. This can help in cases where the
   * CDF calculation involves creating an object. By using this implementation we can make sure
   * only 1 object is created per partition, versus 1 per observation.
   * @param dat the data we wish to evaluate
   * @param distCalc a function to calculate the distance between an empirical value and the
   *                 theoretical value
   * @return the KS statistic and p-value associated with a two sided test
   */
  def testOneSampleOpt(dat: RDD[Double],
                       distCalc: Iterator[(Double, Double, Double)] => Iterator[Double])
    : (Double, Double) = {
    val empiriRDD = ecdf(dat) // empirical distribution
    val distances = empiriRDD.mapPartitions(distCalc, false)
    val ksStat = distances.max
    evalP(ksStat, distances.count())
  }

  /**
   * @return Return a function that we can map over partitions to calculate the KS distance in each
   */
  def stdNormDistances(): (Iterator[(Double, Double, Double)]) => Iterator[Double] = {
    val dist = new NormalDistribution(0, 1)
    (part: Iterator[(Double, Double, Double)]) => part.map {
      case (v, dl, dp) => {
        val cdfVal = dist.cumulativeProbability(v)
        Math.max(cdfVal - dl, dp - cdfVal)
      }
    }
  }

  /**
   * A convenience function that allows running the KS test for 1 set of sample data against
   * a named distribution
   * @param dat the sample data that we wish to evaluate
   * @param distName the name of the theoretical distribution
   * @return The KS statistic and p-value associated with a two sided test
   */
  def testOneSample(dat: RDD[Double], distName: String): (Double, Double) = {
    val distanceCalc =
      distName match {
        case "stdnorm" => stdNormDistances()
        case  _ =>  throw new UnsupportedOperationException()
      }

    testOneSampleOpt(dat, distanceCalc)
  }

  private[KS] def evalP(ksStat: Double, n: Long): (Double, Double) = {
    val pval = 1 - new KolmogorovSmirnovTest().cdf(ksStat, n.toInt)
    (ksStat, pval)
  }

}
