/*
 A first cut draft in terms of what the KS test implementation might look like 
*/

package com.cloudera.sparkprojects
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest

import org.apache.spark.rdd.RDD
import Math._

object KS {
  def ecdf(dat: RDD[Double]): RDD[(Double, Double)] = {
    val n = dat.count().toDouble
    dat.sortBy(x => x).zipWithIndex().map { case (v, i) => (v, i / n) }
  }

  // KS test for 1 set of sample data (versus theoretical distribution)
  // 2-sided variant
  def testOneSample(dat: RDD[Double], cdf: Double => Double): (Double, Double) = {
    val empiriRDD = ecdf(dat) // empirical distribution
    val distances = empiriRDD.map { case (v, empVal) => Math.abs(cdf(v) -  empVal) }
    val ksStat = distances.max
    evalOneSided(ksStat, distances.count())
  }

  // more efficient implementation for partitions
  def testOneSample(dat: RDD[Double], partOp: Iterator[(Double, Double)] => Iterator[Double])
    : (Double, Double) = {
    val empiriRDD = ecdf(dat) // empirical distribution
    val distances = empiriRDD.mapPartitions(partOp, false)
    val ksStat = distances.max
    evalOneSided(ksStat, distances.count())
  }

  // Create only 1 distribution object per partition (rather than 1 per observation)
  def stdNormMap(): (Iterator[(Double, Double)]) => Iterator[Double] = {
    val dist = new NormalDistribution(0, 1)
    (part: Iterator[(Double, Double)]) => part.map {
      case (v, empVal) => Math.abs(dist.cumulativeProbability(v) - empVal)
    }
  }

  // KS test for 1 set of sample data, versus theoretical, using string for name
  // of distribution
  def testOneSample(dat: RDD[Double], distName: String): (Double, Double) = {
    val cdf =
      distName match {
        case "stdnorm" => stdNormMap()
        case  _ =>  throw new UnsupportedOperationException()
      }

    testOneSample(dat, cdf)
  }


  def testTwoSample(dat1: RDD[Double], dat2: RDD[Double]): (Double, Double) = { 
    if(dat1.count != dat2.count) {
      // throw exception for mismatched lengths (search mllib for existing exception)
      throw new Exception()
    }
    val empiri1 = ecdf(dat1)
    val empiri2 = ecdf(dat2)
    val distances = empiri1.zip(empiri2).map { case (p1, p2) => Math.abs(p1._2 - p2._2) }
    val ksStat = distances.max
    evalOneSided(ksStat, distances.count())
  }
  
  private[KS] def evalOneSided(ksStat: Double, n: Long): (Double, Double) = {
    val pval = 1 - new KolmogorovSmirnovTest().cdf(ksStat, n.toInt)
    (ksStat, pval)
  }

}
