add_mlir_library(FlashTransforms
  RegisterPasses.cpp               # <--- the new file
  DEPENDS
    MLIRFlashOpsIncGen
    MLIRFlashTransformsPassIncGen
    
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
)

add_subdirectory(GraphOptimzations)

