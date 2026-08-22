# Matched Selection-Q Feedback

**Classification:** `root_q_feedback_dominant`

**Recommended follow-up:** Test root-Q confidence/shrinkage rather than modifying the value network.

## Primary Outcome

```json
{
  "all_qsync_32": {
    "rescue_rate": {
      "estimate": 0.025,
      "lower_95": 0.0,
      "samples": 10000,
      "upper_95": 0.075
    },
    "rescued_root_hashes": [
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4"
    ],
    "retained_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
      "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
      "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
      "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674",
      "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
      "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
      "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
      "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
      "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
      "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
      "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
      "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
      "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
      "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
      "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
      "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
      "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
      "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
      "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
      "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
      "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retention_rate": {
      "estimate": 0.975,
      "lower_95": 0.925,
      "samples": 10000,
      "upper_95": 1.0
    }
  },
  "all_qsync_rest": {
    "rescue_rate": {
      "estimate": 0.975,
      "lower_95": 0.925,
      "samples": 10000,
      "upper_95": 1.0
    },
    "rescued_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
      "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
      "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
      "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
      "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
      "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
      "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
      "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
      "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
      "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
      "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
      "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
      "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
      "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
      "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
      "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
      "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
      "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
      "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
      "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retained_root_hashes": [
      "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674"
    ],
    "retention_rate": {
      "estimate": 0.025,
      "lower_95": 0.0,
      "samples": 10000,
      "upper_95": 0.075
    }
  },
  "root_qsync": {
    "rescue_rate": {
      "estimate": 0.975,
      "lower_95": 0.925,
      "samples": 10000,
      "upper_95": 1.0
    },
    "rescued_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
      "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
      "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
      "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
      "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
      "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
      "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
      "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
      "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
      "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
      "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
      "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
      "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
      "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
      "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
      "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
      "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
      "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
      "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
      "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retained_root_hashes": [
      "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674"
    ],
    "retention_rate": {
      "estimate": 0.025,
      "lower_95": 0.0,
      "samples": 10000,
      "upper_95": 0.075
    }
  }
}
```

## Frozen 40-Root Rescue Table

| Root hash | root_qsync | all_qsync_32 | all_qsync_rest |
| --- | --- | --- | --- |
| `013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6` | rescued | retained | rescued |
| `02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e` | rescued | retained | rescued |
| `15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6` | rescued | retained | rescued |
| `188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd` | rescued | retained | rescued |
| `2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5` | rescued | retained | rescued |
| `2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0` | rescued | retained | rescued |
| `2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3` | rescued | retained | rescued |
| `2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864` | rescued | retained | rescued |
| `362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674` | retained | retained | retained |
| `37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b` | rescued | retained | rescued |
| `4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697` | rescued | retained | rescued |
| `6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1` | rescued | retained | rescued |
| `6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e` | rescued | retained | rescued |
| `6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014` | rescued | retained | rescued |
| `700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014` | rescued | retained | rescued |
| `70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2` | rescued | retained | rescued |
| `7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5` | rescued | retained | rescued |
| `8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494` | rescued | retained | rescued |
| `934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267` | rescued | retained | rescued |
| `9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4` | rescued | rescued | rescued |
| `951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b` | rescued | retained | rescued |
| `966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4` | rescued | retained | rescued |
| `a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b` | rescued | retained | rescued |
| `a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01` | rescued | retained | rescued |
| `a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7` | rescued | retained | rescued |
| `afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61` | rescued | retained | rescued |
| `b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7` | rescued | retained | rescued |
| `bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41` | rescued | retained | rescued |
| `c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4` | rescued | retained | rescued |
| `cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055` | rescued | retained | rescued |
| `ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc` | rescued | retained | rescued |
| `d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16` | rescued | retained | rescued |
| `d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90` | rescued | retained | rescued |
| `e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438` | rescued | retained | rescued |
| `e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e` | rescued | retained | rescued |
| `ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf` | rescued | retained | rescued |
| `ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5` | rescued | retained | rescued |
| `f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958` | rescued | retained | rescued |
| `fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709` | rescued | retained | rescued |
| `fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812` | rescued | retained | rescued |

## Negative Controls

```json
{
  "all_qsync_32": 0,
  "all_qsync_rest": 0,
  "full": 0,
  "new_divergences_all_qsync_rest": 0,
  "root_qsync": 0
}
```

| Root hash | full | root_qsync | all_qsync_32 | all_qsync_rest |
| --- | --- | --- | --- | --- |
| `de27e90f400c912ca445b9b03130a3615a243a7e2b03d1ff0d2ff8876553b0c5` | stable | stable | stable | stable |
| `ef9a24395dc93c13c0e07065c36eba08e140414e3fbcb4094af83ed34557184e` | stable | stable | stable | stable |
| `336acfa7ded673514ad4ef1c0e56dfd6c4d45b8953c3a269147e3fd091b70e4f` | stable | stable | stable | stable |
| `fb1a08a6157b3eb59ad341976df7be2ea75e69cc713c98b707444493dbd76d24` | stable | stable | stable | stable |
| `096898d2d88363e213b36771398db724afd6a5a3571346826046bc32476354d4` | stable | stable | stable | stable |
| `4b9792cd7cae924c397ba85ca5e912597884c14f6284e78516245b69eeac9af3` | stable | stable | stable | stable |
| `fd51a92ff90680c9a8761731bb182ccee58b84e06abad382a11bdabbe5951a5c` | stable | stable | stable | stable |
| `553b183d8ca98e641a07b256798fe7c48b34b0718b2a34e9eb92bb68703cf789` | stable | stable | stable | stable |
| `0347d2b31667eb751e1dbb9d42097fd29b3e4507a447da7b0cd735fada78de23` | stable | stable | stable | stable |
| `e3ee4b4a73a9f15e1abb8cbaa47a240e590621e8c4157957513b513e331197a9` | stable | stable | stable | stable |
| `a755d4c554746f6ea147d78961417e9d557c6f5da12fe0f1502a780c971a94df` | stable | stable | stable | stable |
| `aaca3ee760cf8f67c9ab04a62d3013d974f8d0cc211bd2b8035510cd7cfd693f` | stable | stable | stable | stable |
| `07c2452705d43679b3f862dfc321aeb16e2e7f8d20893e94e4f6a37dc1ddda40` | stable | stable | stable | stable |
| `0c948430b56bdef88f601bc0fb879f5bdef18b285a03ecbb2918a607ee6c8143` | stable | stable | stable | stable |
| `f21473a7346ae37144c86322779edb8aefa55d827e51b30a238303e6704ed2f1` | stable | stable | stable | stable |
| `920419c02b6da607f91fd3e3d14bfbb3493958d149390fe6f4bd719985776ff9` | stable | stable | stable | stable |
| `8676fc0442593252a55925d4689a22c324229cd19f0f8632437d1b1a9311642d` | stable | stable | stable | stable |
| `cfd31c47ba43f707c03d1bda3fef92bea24a36237b9d73c81caf330686c45b93` | stable | stable | stable | stable |
| `cc3617e8863481a8089c4e6675a747d8c85e56496f19c4f4e2072c2b25d9897e` | stable | stable | stable | stable |
| `fc5bf26ecc9c3c6475d08f6fa46e43693db3148154dfd3b4ea6a641130312c17` | stable | stable | stable | stable |
| `6f9e1e6fefd6176f3782ac3d5b3ca087a70914589cb10bbb5d6a53baa8d37742` | stable | stable | stable | stable |
| `2033250a9f6439989144ec8a14e1bf9a70cce5a08806b6e99b02112f485b7382` | stable | stable | stable | stable |
| `046d95f1e70f67e3fe754292f6f1c8a92019b8a9a4bf10e47f36440e6c89db09` | stable | stable | stable | stable |
| `0f9140d59b5339aded929c5f14ddef7755a16c5b900cd7abd449f59ab7df3b18` | stable | stable | stable | stable |
| `7795a0427a6b447971f77427d756aa74e6eea5c0718b9de113eb28661970ba2e` | stable | stable | stable | stable |
| `19a46f0773e2f100996ff7cd22af1fc4cb1ca6378ae5b66a1428dabbcfdad3d0` | stable | stable | stable | stable |
| `3330531f636169691927faaaabc4b2e68f04450c91a88439f29023420562c269` | stable | stable | stable | stable |
| `f9ed552d9613cc1cdfc3fb4094b0cbd0f901dc59a9f1a91afe4d3f7782a548da` | stable | stable | stable | stable |
| `ead506a1149a0aa405bdc1513e54f6c7a0dc0d603d5bb4b77761e226b0e43089` | stable | stable | stable | stable |
| `8bbf0083453efbccfb3f4f32e431374f8c68ed33ea4db66e0d92a7d3a98826ca` | stable | stable | stable | stable |
| `c2d6250b2fa4c3a1b08b68f36eabd6bc05a368a506800f8c6cd4660228bcf966` | stable | stable | stable | stable |
| `46b9b6e5a1ed0325d5c31630bfb9b26917551b3878f9fd67b89371b473b8f38a` | stable | stable | stable | stable |
| `7239f8b2c6dd4e4c0bd1dc6a73b9d0ef39a5ca6b39337aa0e1419fda4e0f7cba` | stable | stable | stable | stable |
| `342ef91973a3fff82f58e981eb2e6e1b2a26f4c74175b5ad6c20c70efde3705d` | stable | stable | stable | stable |
| `47f2361ec0e33ec141c5585a0d5cf40351333f7ad92de1b1d0c9bb1a22ab0acd` | stable | stable | stable | stable |
| `b9e27e926dc565d8de119af9a35fab79f98d4b360f18dd74b61bb9dec51e8849` | stable | stable | stable | stable |
| `497819c496849da0cac8ee7143ce915b4df1962631bb8e2acf80b11786b58b15` | stable | stable | stable | stable |
| `ed67a8db689cf12a42705036853e917fc3a79084f329f3218f57b0504e4f09b7` | stable | stable | stable | stable |
| `f89ef146c48e3bbb02c68084b52330c6e2759b1d8527c2555cf0c1073a099fe2` | stable | stable | stable | stable |
| `54dc994804cd7e6093e85de20fdd885993105a182f5350733d1fb57db2085c5a` | stable | stable | stable | stable |

## Coverage Telemetry

```json
{
  "all_qsync_32": {
    "applied_selection_q": 27254,
    "eligible_selection_edges": 28100,
    "missing_reference": 770,
    "missing_reference_action": 1,
    "p1_unvisited_skips": 75
  },
  "all_qsync_rest": {
    "applied_selection_q": 1771853,
    "eligible_selection_edges": 1929056,
    "missing_reference": 154141,
    "missing_reference_action": 136,
    "p1_unvisited_skips": 2926
  },
  "reference_capture": {
    "candidate_nodes": 38424787,
    "duplicate_state_hashes": 3407298,
    "non_equivalent_duplicate_state_hashes": 2774452,
    "reference_snapshots": 31574100
  },
  "root_qsync": {
    "applied_selection_q": 397966,
    "eligible_selection_edges": 397990,
    "p1_unvisited_skips": 24
  }
}
```

## Invariants

```json
{
  "all_selection_q_lane_invariants": true,
  "artifact_hashes": true,
  "frozen_identity_hash": true,
  "full_amplified_reproduces": true,
  "full_first_divergences_reproduce": true,
  "replay_hash": true,
  "state_hashes": true
}
```
