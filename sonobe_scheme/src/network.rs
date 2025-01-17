use ark_bn254::Fr;
use serde::{Deserialize, Deserializer};
use serde::de::{self, Visitor, MapAccess};
use std::fmt;
use std::str::FromStr;
#[derive(Debug)]
pub struct DenseLayer {
    pub weight: Vec<Vec<Fr>>, // dims: [nInputs x nOutput]
    pub bias: Vec<Fr>,        // dims: [nOutputs]
    pub dense_out: Vec<Fr>,   // dims: [nOutputs]
    pub remainder: Vec<Fr>,   // dims: [nOutputs]
    pub activation: Vec<Fr>,  // dims: [nOutputs]
}

#[derive(Debug)]
pub struct TailLayer {
    pub weight: Vec<Vec<Fr>>, // dims: [nInputs x nOutput]
    pub bias: Vec<Fr>,        // dims: [nOutputs]
    pub dense_out: Vec<Fr>,   // dims: [nOutputs]
    pub remainder: Vec<Fr>,   // dims: [nOutputs]
    pub activation: Fr,       // dims: [nOutputs]
}

#[derive(Debug)]
pub struct Network {
    pub x: Vec<Fr>,               // dims: [nRows x nCols]
    pub head: DenseLayer,         // Head of the network
    pub backbone: Vec<DenseLayer>, // Backbone layers
    pub tail: TailLayer,          // Tail layer
}

// Custom deserialization for Vec<Fr>
fn deserialize_vec_fr<'de, D>(deserializer: D) -> Result<Vec<Fr>, D::Error>
where
    D: Deserializer<'de>,
{
    let str_vec: Vec<String> = Vec::deserialize(deserializer)?;
    str_vec
        .into_iter()
        .map(|s| {
            Fr::from_str(&s).map_err(|_| {
                de::Error::custom(format!("Failed to parse Fr from string: {}", s))
            })
        })
        .collect()
}

// Implement Deserialize for DenseLayer
impl<'de> Deserialize<'de> for DenseLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DenseLayerInternal {
            weight: Vec<Vec<String>>,
            bias: Vec<String>,
            dense_out: Vec<String>,
            remainder: Vec<String>,
            activation: Vec<String>,
        }

        let internal = DenseLayerInternal::deserialize(deserializer)?;

        Ok(DenseLayer {
            weight: internal
                .weight
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|s| {
                            Fr::from_str(&s).map_err(|_| {
                                de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                            })
                        })
                        .collect()
                })
                .collect::<Result<Vec<Vec<Fr>>, _>>()?,
            bias: internal
                .bias
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            dense_out: internal
                .dense_out
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            remainder: internal
                .remainder
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            activation: internal
                .activation
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
        })
    }
}

// Implement Deserialize for TailLayer
impl<'de> Deserialize<'de> for TailLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TailLayerInternal {
            weight: Vec<Vec<String>>,
            bias: Vec<String>,
            dense_out: Vec<String>,
            remainder: Vec<String>,
            activation: String,
        }

        let internal = TailLayerInternal::deserialize(deserializer)?;

        Ok(TailLayer {
            weight: internal
                .weight
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|s| {
                            Fr::from_str(&s).map_err(|_| {
                                de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                            })
                        })
                        .collect()
                })
                .collect::<Result<Vec<Vec<Fr>>, _>>()?,
            bias: internal
                .bias
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            dense_out: internal
                .dense_out
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            remainder: internal
                .remainder
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            activation: Fr::from_str(&internal.activation).map_err(|_| {
                de::Error::custom(format!(
                    "Failed to parse Fr from string: {}",
                    internal.activation
                ))
            })?,
        })
    }
}

// Implement Deserialize for Network
impl<'de> Deserialize<'de> for Network {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NetworkInternal {
            x: Vec<String>,
            head: DenseLayer,
            backbone: Vec<DenseLayer>,
            tail: TailLayer,
        }

        let internal = NetworkInternal::deserialize(deserializer)?;

        Ok(Network {
            x: internal
                .x
                .into_iter()
                .map(|s| {
                    Fr::from_str(&s).map_err(|_| {
                        de::Error::custom(format!("Failed to parse Fr from string: {}", s))
                    })
                })
                .collect::<Result<Vec<Fr>, _>>()?,
            head: internal.head,
            backbone: internal.backbone,
            tail: internal.tail,
        })
    }
}
