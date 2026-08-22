# Synchronized Backup Localization of Post-Divergence Q Feedback

**Classification:** `intervention_creates_new_instability`

**Recommended follow-up:** Use matched-node Q synchronization instead.

## Primary Outcome

```json
{
  "sync_1": {
    "rescue_rate": {
      "estimate": 0.15,
      "lower_95": 0.05,
      "samples": 10000,
      "upper_95": 0.275
    },
    "rescued_root_hashes": [
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709"
    ],
    "retained_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
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
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retention_rate": {
      "estimate": 0.85,
      "lower_95": 0.725,
      "samples": 10000,
      "upper_95": 0.95
    }
  },
  "sync_128": {
    "rescue_rate": {
      "estimate": 0.45,
      "lower_95": 0.3,
      "samples": 10000,
      "upper_95": 0.6
    },
    "rescued_root_hashes": [
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
      "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
      "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
      "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
      "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
      "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
      "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
      "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retained_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
      "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
      "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674",
      "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
      "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
      "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
      "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
      "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
      "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
      "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
      "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
      "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958"
    ],
    "retention_rate": {
      "estimate": 0.55,
      "lower_95": 0.4,
      "samples": 10000,
      "upper_95": 0.7
    }
  },
  "sync_32": {
    "rescue_rate": {
      "estimate": 0.25,
      "lower_95": 0.125,
      "samples": 10000,
      "upper_95": 0.4
    },
    "rescued_root_hashes": [
      "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
      "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
      "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retained_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
      "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
      "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674",
      "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
      "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
      "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
      "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
      "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
      "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
      "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
      "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
      "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
      "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
      "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
      "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
      "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
      "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958"
    ],
    "retention_rate": {
      "estimate": 0.75,
      "lower_95": 0.6,
      "samples": 10000,
      "upper_95": 0.875
    }
  },
  "sync_8": {
    "rescue_rate": {
      "estimate": 0.25,
      "lower_95": 0.125,
      "samples": 10000,
      "upper_95": 0.4
    },
    "rescued_root_hashes": [
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709"
    ],
    "retained_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
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
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
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
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retention_rate": {
      "estimate": 0.75,
      "lower_95": 0.6,
      "samples": 10000,
      "upper_95": 0.875
    }
  },
  "sync_rest": {
    "rescue_rate": {
      "estimate": 0.45,
      "lower_95": 0.3,
      "samples": 10000,
      "upper_95": 0.6
    },
    "rescued_root_hashes": [
      "013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6",
      "15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6",
      "188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd",
      "2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0",
      "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674",
      "37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b",
      "6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e",
      "6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014",
      "7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5",
      "8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494",
      "934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267",
      "9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4",
      "a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b",
      "a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7",
      "b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7",
      "cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055",
      "e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e",
      "ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5"
    ],
    "retained_root_hashes": [
      "02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e",
      "2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5",
      "2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3",
      "2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864",
      "4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697",
      "6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1",
      "700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014",
      "70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2",
      "951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b",
      "966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4",
      "a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01",
      "afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61",
      "bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41",
      "c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4",
      "ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc",
      "d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16",
      "d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90",
      "e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438",
      "ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf",
      "f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958",
      "fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709",
      "fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812"
    ],
    "retention_rate": {
      "estimate": 0.55,
      "lower_95": 0.4,
      "samples": 10000,
      "upper_95": 0.7
    }
  }
}
```

## Minimum Synchronization Window

```json
{
  "first_rescued_by_128": 7,
  "first_rescued_by_32": 5,
  "first_rescued_by_8": 5,
  "never_rescued": 9,
  "requires_rest": 8,
  "rescued_by_1": 6
}
```

## Frozen 40-Root Retention Table

| Root hash | sync_1 | sync_8 | sync_32 | sync_128 | sync_rest | Minimum |
| --- | --- | --- | --- | --- | --- | --- |
| `013807a2efe746bf62ed46e16f05b65680b500b149bbb039c1db9c912f4688e6` | retained | retained | retained | retained | rescued | sync_rest |
| `02ed9872133bcf81ccedc7752b4e45da4eed3e7ddb19b8bd2af2eb1b3a5b4c8e` | retained | retained | retained | rescued | retained | sync_128 |
| `15254d26866a519e19b05ecb7d074efca290aee7abeada5aad969411560f1cb6` | rescued | rescued | retained | retained | rescued | sync_1 |
| `188c503236f7619ab9b08523afef1335dc2a110079e95a6d45866601a67b10cd` | retained | rescued | retained | rescued | rescued | sync_8 |
| `2a9f8e2a5823944f71283c4afc3b75295b470f794cb00d86bf43307dfeb12da5` | rescued | retained | retained | retained | retained | sync_1 |
| `2ea19f706ad5e1a5ce7313af831803e1a54b282b3446d2edbe88262a33fe43f0` | retained | retained | rescued | rescued | rescued | sync_32 |
| `2f05808aa02b8d8cfff5427b0e1412404e9679022172b74e4c9c3035a592a5e3` | retained | retained | retained | retained | retained | never |
| `2f9f8c9a2de6867a3732c41a2c67c896a902dff9bdd621cff95b0257dc62e864` | retained | retained | retained | retained | retained | never |
| `362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674` | retained | retained | retained | retained | rescued | sync_rest |
| `37cd52ab179163de0784b3bc5e242c12e84d7226c58499baa8e01d6050c9356b` | retained | retained | retained | rescued | rescued | sync_128 |
| `4502ce0c8cb224305db955d0b6a7ac0a3b8a757e2d3ac5eeb1b15f24a7e38697` | retained | retained | retained | retained | retained | never |
| `6805f4349f1762bbcd7372428cae1c1687bae8f6a13e1c00354b37d39aeef7a1` | retained | retained | retained | retained | retained | never |
| `6a2bcb7f4757ba4582ab22e17d2d29e11a6502e40c3326308c4d40f4027dbc1e` | retained | retained | retained | rescued | rescued | sync_128 |
| `6d97c9b583700cb29c9156fe1e0d303e7f884272f22a826b04734dc7a4e65014` | retained | retained | retained | retained | rescued | sync_rest |
| `700da2fd2f248840eeb53f501fac491889cf188e44737429c8b228969c253014` | retained | retained | rescued | rescued | retained | sync_32 |
| `70126c0e1da58b32ee60ff96ce9ca8286c1912b22fa3c7c232bb8a847e599ae2` | retained | rescued | rescued | retained | retained | sync_8 |
| `7f9548bfa2270253e9e030f378b1c70a2f03d63429f3fb0f8752f433ca9f01e5` | retained | retained | retained | retained | rescued | sync_rest |
| `8f1d0a37da6fa80727a7d2fd261bf1a54366c273a47019b9a7577f8dce1b6494` | retained | rescued | retained | retained | rescued | sync_8 |
| `934f7962a8d86e237fa09895ba3bca4bb7a0c46902c952262be945914c08b267` | rescued | rescued | rescued | rescued | rescued | sync_1 |
| `9377e1f332ea635ddf69a42542e8752ee65ff25da4af989589e0eec759d242f4` | rescued | rescued | rescued | rescued | rescued | sync_1 |
| `951a3042f34f40919ac5a5eb66a7072029979b6f604f11e37c733fd28471258b` | rescued | rescued | retained | retained | retained | sync_1 |
| `966a611e92bed9435bb1c67a58eeaea3d87df8959291b481aca509e9f023b8e4` | retained | retained | retained | retained | retained | never |
| `a15239440233e6fa063c7a2dfd1c140fdd2df2f89c72efaf35ee409cf8eb146b` | retained | retained | retained | retained | rescued | sync_rest |
| `a482b39e3d6943f9a7f1c360fa8dd1177ff89dbede16049136ab46032640ed01` | retained | retained | retained | retained | retained | never |
| `a6a0b90f828281f037127400b801dfd1f92611669b29fafeabecd5dac29a5ef7` | retained | retained | retained | retained | rescued | sync_rest |
| `afe40174ccfd7dc7a5de76bf3753da9bc6ebaf8ce75f25b2f7241c450ffe1d61` | retained | retained | retained | rescued | retained | sync_128 |
| `b94cfa81d691473293345c4e31ba97e3fe017e2fb0818ed5b3bab17c4a4835d7` | retained | retained | retained | retained | rescued | sync_rest |
| `bd815c9c00d4e0eac19b49257928c4820d6e54fa21f62f62fa99bff31209cc41` | retained | retained | retained | rescued | retained | sync_128 |
| `c77c01ee1fffa63b9ae1b1df4d1dea07485efa23da81b76221e7aed7781184b4` | retained | retained | rescued | rescued | retained | sync_32 |
| `cd86d5db65feb065b02f2c696b2598934922f81874e632ae14946eaa0f522055` | retained | retained | retained | retained | rescued | sync_rest |
| `ce1a40b6c89e5b666369ea705346f426e430647c0c39c1c4488412d0b410aefc` | retained | retained | rescued | rescued | retained | sync_32 |
| `d4a9f78ae368a2e5d0edbd2f241d359dca4f03efb7b3891cf895d94d155a8d16` | retained | retained | retained | rescued | retained | sync_128 |
| `d89fa6df131c62b8415bff3bd444badbd6867f5be484f45dda5973b282373e90` | retained | retained | retained | retained | retained | never |
| `e3c5f232ebd455665d7c89a94884c33c61e3421104dfb8bfd79418968ff91438` | retained | rescued | rescued | rescued | retained | sync_8 |
| `e6f397b4ede2144309468114e995d1843c15770773058e45ae644576222cb40e` | retained | rescued | retained | rescued | rescued | sync_8 |
| `ed1b01e0f0e8c9d1f7a92485669235b6bc4369fe793996c6a6840e319b0260bf` | retained | retained | retained | retained | retained | never |
| `ee50f0a3b7cbf15433705ee08e3eb6bde775b453374eb66e16e30c1323c73ab5` | retained | retained | retained | rescued | rescued | sync_128 |
| `f4955ba6a6e86f529ea6398a94d0f739510de1778cead33cf3302e54b4a48958` | retained | retained | retained | retained | retained | never |
| `fdec50d74542bcb4dd393e97be2ec07ef25ef6e12444c753cc4d3abf0f9f0709` | rescued | rescued | rescued | rescued | retained | sync_1 |
| `fe844ca60fd0c8df245ad384a4041488a5be4f08c48d1d06613b553b60cb0812` | retained | retained | rescued | rescued | retained | sync_32 |

## Invariants

```json
{
  "all_sync_invariants": true,
  "artifact_hashes": true,
  "frozen_identity_hash": true,
  "full_amplified_reproduces": true,
  "full_first_divergences_reproduce": true,
  "replay_hash": true,
  "state_hashes": true
}
```

## Negative Controls

```json
{
  "full": 0,
  "new_divergences_sync_rest": 22,
  "sync_32": 0,
  "sync_rest": 22
}
```
