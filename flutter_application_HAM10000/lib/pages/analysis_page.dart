import 'dart:io';
import 'dart:math' as dart_math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:screenshot/screenshot.dart';
import 'package:share_plus/share_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import '../database_helper.dart';

class AnalysisPage extends StatefulWidget {
  const AnalysisPage({super.key});

  @override
  State<AnalysisPage> createState() => _AnalysisPageState();
}

class _AnalysisPageState extends State<AnalysisPage> {
  final ImagePicker _picker = ImagePicker();
  ClassificationModel? _imageModel;
  ClassificationModel? _binaryModel;
  double _binaryThreshold = 0.69;
  XFile? _imageFile;
  bool _isLoading = false;
  
  List<MapEntry<String, double>> _topResults = [];
  bool _underConfidence = false;
  
  String? _camImageBase64;
  bool _isExplaining = false;
  bool _isSaved = false;
  
  final ScreenshotController _screenshotController = ScreenshotController();
  final DatabaseHelper _dbHelper = DatabaseHelper();
  final Uuid _uuid = const Uuid();

  final Map<String, String> _riskLevels = {
    'akiec': 'Yüksek Risk',
    'bcc': 'Yüksek Risk',
    'mel': 'Yüksek Risk',
    'bkl': 'Düşük Risk',
    'df': 'Düşük Risk',
    'nv': 'Düşük Risk',
    'vasc': 'Orta Risk'
  };

  final Map<String, Color> _riskColors = {
    'akiec': Colors.redAccent,
    'bcc': Colors.redAccent,
    'mel': Colors.redAccent,
    'bkl': Colors.greenAccent,
    'df': Colors.greenAccent,
    'nv': Colors.greenAccent,
    'vasc': Colors.orangeAccent
  };

  final Map<String, String> _classNames = {
    'akiec': 'Aktinik Keratoz (AKIEC)',
    'bcc': 'Bazal Hücreli Karsinom (BCC)',
    'bkl': 'Benign Keratoz (BKL)',
    'df': 'Dermatofibrom (DF)',
    'mel': 'Melanom (MEL)',
    'nv': 'Melanositik Nevüs (NV)',
    'vasc': 'Vasküler Lezyon (VASC)'
  };

  final List<String> _labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];

  final Map<String, String> _classExplanations = {
    'akiec': 'Model, lezyonun pembe/kırmızımsı bir zemin üzerine yerleştiğini ve yüzeyinde ince beyaz pullanmalar (kepeklenmeler) olduğunu fark etti. Özellikle gözeneklerin etrafındaki "çilek deseni" (strawberry pattern) adı verilen bu doku, güneş hasarına bağlı oluşan Aktinik Keratoz ile uyumludur. Kırmızı alanlar, modelin bu pullu ve pütürlü zemin yapısına odaklandığı yerlerdir.',
    'bcc': 'Model, lezyonun yüzeyinde ağaç dalına benzeyen belirgin kırmızı kılcal damarlar (arborizing vessels) ve mavimsi/gri ufak noktacıklar tespit etti. Bu parlak ve pembemsi görünüm Bazal Hücreli Karsinom\'un en tipik imzasıdır. Kırmızı alanlar, modelin kanseri teşhis ederken bu anormal damarlanmalara ve gri noktalara dikkat ettiğini göstermektedir.',
    'bkl': 'Model, lezyonun üzerinde delik delik siyah/koyu kahverengi noktacıklar (komedo benzeri tıkaçlar) gördü. Sanki derinin üzerine sonradan yapıştırılmış gibi duran bu pürüzlü ve keskin sınırlı yapı, iyi huylu bir yaşlılık lekesi olan Seboreik Keratoz ile tam uyumludur. Kırmızı bölgeler, modelin bu zararsız tıkaçları yakaladığı yerlerdir.',
    'df': 'Model, lezyonun tam ortasında sanki eskiden kesilip iyileşmiş gibi duran parlak beyaz bir yara izi (central white patch) tespit etti. Bu beyazlığın etrafını saran ince kahverengi ağ yapısı, Dermatofibrom adı verilen iyi huylu sert nodüllerin en belirgin özelliğidir. Isı haritasındaki kırmızı yoğunluk, modelin doğrudan bu ortadaki yara izi (skar) desenini hedef aldığını gösterir.',
    'mel': 'Model, bu lezyonda ciddi şüpheli bulgular tespit etti:\n- Kötü huylu asimetri (şekil bozukluğu)\n- Düzensiz ve haritaya benzeyen kenar sınırları\n- Siyah, mavi, kahverengi ve kırmızının karıştığı çoklu renk cümbüşü\n- Yüzeyde puslu bir mavi-beyaz örtü (blue-white veil)\n\nKırmızı alanlar, modelin kanser kararı verirken bu tehlikeli düzensizlikleri ve renk asimetrisini yakaladığı odak noktalarıdır. Lütfen acilen bir dermatoloğa başvurun.',
    'nv': 'Model, bu lezyonun kusursuz bir yuvarlaklıkta (simetrik) olduğunu ve renginin tamamen homojen (tek tip kahverengi) dağıldığını tespit etti. Keskin kenar sınırları ve düzenli yapısı sayesinde bunun tamamen zararsız bir normal ben (Nevüs) olduğuna karar verdi. Kırmızı alanlar, modelin bu pürüzsüz ve düzenli simetriyi onayladığı bölgelerdir.',
    'vasc': 'Model, lezyonun kahverengi veya siyah değil, doğrudan kan renginde (kırmızı, mor veya bordo) olduğunu tespit etti. Böğürtlen taneleri gibi yan yana dizilmiş yuvarlak "kan gölcükleri" (lacunae) yapısı, bunun damar kaynaklı zararsız bir Vasküler Lezyon (örn. Hemanjiyom) olduğunu kanıtlar. Kırmızı bölgeler, modelin bu kan gölcüklerine odaklandığı yerlerdir.'
  };
  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _imageModel = await PytorchLite.loadClassificationModel(
        "assets/models/model.pt", 
        260, 
        260, 
        7
      );

      _binaryModel = await PytorchLite.loadClassificationModel(
        "assets/models/lesion_binary_model_torchscript.pt",
        224,
        224,
        2, // numberOfClasses: lezyon_degil, lezyon
      );

      try {
        final thresholdJson = await rootBundle.loadString("assets/models/lesion_binary_threshold.json");
        final thresholdData = json.decode(thresholdJson);
        _binaryThreshold = (thresholdData['threshold'] ?? 0.69).toDouble();
      } catch (e) {
        debugPrint("Threshold okuma hatası: $e");
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text("Model yükleme hatası: $e"),
        ));
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    // Show flashlight tip before opening camera
    if (source == ImageSource.camera && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("📸 İpucu: En doğru analiz için kameranızın flaşını açarak çekim yapın!"),
          duration: Duration(seconds: 4),
          backgroundColor: Colors.teal,
        )
      );
    }

    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      CroppedFile? croppedFile = await ImageCropper().cropImage(
        sourcePath: pickedFile.path,
        uiSettings: [
          AndroidUiSettings(
            toolbarTitle: 'Yalnızca Lekeyi Seçin',
            toolbarColor: Colors.teal,
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: false,
            hideBottomControls: false,
          ),
          IOSUiSettings(
            title: 'Yalnızca Lekeyi Seçin',
            aspectRatioLockEnabled: false,
          ),
        ],
      );

      if (croppedFile != null) {
        setState(() {
          _imageFile = XFile(croppedFile.path);
          _isLoading = true;
          _topResults.clear();
          _underConfidence = false;
          _camImageBase64 = null;
          _isSaved = false;
        });
        await _analyzeImage();
      }
    }
  }

  Future<void> _analyzeImage({bool forcePhase2 = false}) async {
    if (_imageFile == null) {
      setState(() { _isLoading = false; });
      return;
    }

    if (_imageModel == null || _binaryModel == null) {
      setState(() { _isLoading = false; });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text("⚠️ Modeller henüz yüklenmedi! Lütfen uygulamayı tamamen kapatıp açın veya Hot Restart yapın."),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
      return;
    }

    try {
      final bytes = await _imageFile!.readAsBytes();

      if (!forcePhase2) {
        // --- AŞAMA 1: Lezyon Algılama (Binary Model) ---
      List<double>? binaryPredictions = await _binaryModel!.getImagePredictionList(
        bytes,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
      );

      if (binaryPredictions.isEmpty) {
        throw Exception("Lezyon tespit modeli boş çıktı verdi.");
      }

      double binaryProb = 0.0;
      String binaryResult = 'lezyon_degil';

      if (binaryPredictions.length == 1) {
        // Model tek çıkış veriyor.
        // Raw logit değerleri:
        //   Bilgisayar/cansız nesne: ~-14 (daha az negatif)
        //   Boş insan derisi: ~-14 civarı (daha az negatif)
        //   Gerçek lezyon: ~-29 (çok daha negatif)
        // Sigmoid her iki değeri de ~0'a çektiği için ayırt edici gücünü kaybediyor.
        // Bu yüzden raw logit değerini doğrudan kullanıyoruz.
        double rawLogit = binaryPredictions[0];
        double rawThreshold = -15.0; // Bilgisayar(-14.7) engellenir, galeriden lezyon(-16.47) geçer

        if (rawLogit < rawThreshold) {
          // Çok negatif = kesinlikle lezyon
          binaryResult = 'lezyon';
          binaryProb = 1.0; // Yüksek güven
        } else {
          // Az negatif veya pozitif = lezyon değil (bilgisayar, boş deri vs.)
          binaryResult = 'lezyon_degil';
          binaryProb = 0.0;
        }
      } else {
        // İki çıkış: softmax uygula
        double maxVal = binaryPredictions.reduce(dart_math.max);
        List<double> exps = binaryPredictions.map((e) => dart_math.exp(e - maxVal)).toList();
        double sumExps = exps.reduce((a, b) => a + b);
        List<double> probs = exps.map((e) => sumExps > 0 ? e / sumExps : 0.0).toList();
        
        // 'lezyon' sınıfının indeksi = 1 (labels: 0=lezyon_degil, 1=lezyon)
        binaryProb = probs[1];
        binaryResult = binaryProb >= _binaryThreshold ? 'lezyon' : 'lezyon_degil';
      }

      debugPrint('--- ÇİFT AŞAMALI AKIŞ LOGLARI ---');
      debugPrint('Binary Predictions: $binaryPredictions');
      debugPrint('Lezyon Olasılığı: $binaryProb');
      debugPrint('Eşik Değeri (Threshold): $_binaryThreshold');
      debugPrint('Karar: $binaryResult');
      debugPrint('----------------------------------');

      if (binaryResult == 'lezyon_degil') {
        setState(() {
          _isLoading = false;
          _topResults.clear();
          _underConfidence = false;
        });

        if (mounted) {
          showDialog(
            context: context,
            barrierDismissible: false,
            builder: (context) => AlertDialog(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              title: const Row(
                children: [
                  Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 28),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      "Lezyon\nAlgılanamadı", 
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
              content: SingleChildScrollView(
                child: Text(
                  "Yapay zeka bu fotoğrafta belirgin bir cilt lezyonu (leke, ben vb.) tespit edemedi.\n\n"
                  "🛠️ Hata Ayıklama Logu:\n"
                  "Raw Değer: ${binaryPredictions.length == 1 ? binaryPredictions[0].toStringAsFixed(2) : binaryProb.toStringAsFixed(2)}\n\n"
                  "Lütfen:\n"
                  "• Yalnızca şüpheli lezyona odaklanın.\n"
                  "• Net, gölgesiz ve iyi ışıkta çekilmiş bir fotoğraf kullanın.\n"
                  "• Arka planda başka cisimler olmamasına dikkat edin.",
                  style: const TextStyle(height: 1.4),
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    setState(() {
                      _isLoading = true;
                    });
                    _analyzeImage(forcePhase2: true);
                  },
                  child: const Text("Yine de Analiz Et", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: Colors.blue)),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    setState(() {
                      _imageFile = null;
                    });
                  },
                  child: const Text("Tamam", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                )
              ],
            ),
          );
        } // closes if (mounted)
        return;
      } // closes if (binaryResult == 'lezyon_degil')
    } // closes if (!forcePhase2)

      // --- AŞAMA 2: Cilt Hastalığı Sınıflandırması (7 Sınıflı Model) ---
      List<double> rawPredictions = await _imageModel!.getImagePredictionList(bytes);
      
      double maxLogit = -double.maxFinite;
      for (var val in rawPredictions) {
         if (!val.isNaN && val > maxLogit) maxLogit = val;
      }
      
      double sumExp = 0.0;
      List<double> expVals = [];
      for (var val in rawPredictions) {
         if (val.isNaN) val = -100.0; 
         double e = dart_math.exp(val - maxLogit);
         expVals.add(e);
         sumExp += e;
      }
      
      List<MapEntry<String, double>> mappedResults = [];
      for (int i = 0; i < _labels.length; i++) {
        double prob = sumExp > 0 ? (expVals[i] / sumExp) : 0.0;
        mappedResults.add(MapEntry(_labels[i], prob));
      }

      mappedResults.sort((a, b) => b.value.compareTo(a.value));

      setState(() {
        _isLoading = false;
        if (mappedResults.first.value < 0.40) {
          _underConfidence = true;
        } else {
          _underConfidence = false;
          _topResults = mappedResults.take(3).toList();
        }
      });

    } catch (e) {
      debugPrint("Analiz Hatası: $e");
      setState(() { _isLoading = false; });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Analiz sırasında bir hata oluştu: $e")),
        );
      }
    }
  }

  Future<void> _saveToDatabase(String bodyPart) async {
    final highest = _topResults.first;
    await _dbHelper.insertLesion({
      'id': _uuid.v4(),
      'image_path': _imageFile!.path,
      'body_part': bodyPart,
      'highest_risk_label': highest.key,
      'risk_level': _riskLevels[highest.key]!,
      'confidence': highest.value * 100,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });
    
    if (mounted) {
      setState(() { _isSaved = true; });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Geçmişe başarıyla kaydedildi!"), backgroundColor: Colors.teal,)
      );
    }
  }

  Future<void> _explainResult() async {
    if (_imageFile == null) return;
    setState(() { _isExplaining = true; });

    try {
      // 10.0.2.2 emülatör içindir. Fiziksel cihazınızın bu bilgisayara bağlanabilmesi için
      // bilgisayarın Wi-Fi yerel IP adresini kullanıyoruz: 10.200.60.204
      var request = http.MultipartRequest('POST', Uri.parse('http://10.134.160.153:8000/explain'));
      request.files.add(await http.MultipartFile.fromPath('file', _imageFile!.path));
      
      var response = await request.send();
      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var jsonResponse = jsonDecode(responseData);
        if (jsonResponse['success'] == true) {
          setState(() {
            _camImageBase64 = jsonResponse['heatmap_base64'];
          });
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text("Açıklama alınamadı: $e")));
      }
    } finally {
      if (mounted) setState(() { _isExplaining = false; });
    }
  }

  Future<void> _shareScreenshot() async {
    try {
      final directory = (await getApplicationDocumentsDirectory()).path;
      
      final imageFile = await _screenshotController.captureAndSave(
        directory, 
        fileName: 'doktor_raporu_${DateTime.now().millisecondsSinceEpoch}.png',
        pixelRatio: 2.0,
      );
      
      if (imageFile != null) {
        String msg = "Merhaba Doktor, telefonumdaki bir yapay zeka uygulamasının yaptığı cilt lezyonu analizini ve o bölgenin fotoğrafını ekte size gönderiyorum.\n\n"
                     "Yapay zekanın tıbbi bir cihaz olmadığını ve pek çok hata yapabileceğini biliyorum. Bu yüzden paniğe kapılmadan önce, fotoğrafı ve durumu profesyonel bir göz olarak kesinlikle sizin değerlendirmenizi ve tıbbi fikrinizi almak istiyorum.";

        await Share.shareXFiles(
          [XFile(imageFile)],
          text: msg,
        );
      }
    } catch (e) {
      print("Screenshot error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Theme.of(context).colorScheme.surface,
            Theme.of(context).colorScheme.surfaceContainerHighest.withOpacity(0.3),
          ],
        ),
      ),
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
               _buildCropWarningBox(),
               const SizedBox(height: 24),
               
               // Wrap both image and results in ScreenshotController
               Screenshot(
                 controller: _screenshotController,
                 child: Container(
                   color: Colors.transparent, 
                   child: Column(
                     children: [
                       _buildImagePreview(),
                       const SizedBox(height: 28),
                       if (_imageFile != null) _buildResultsSection(),
                     ],
                   )
                 ),
               ),
               
               const SizedBox(height: 32),
               if (_imageFile == null) _buildControlButtons(),
               if (_imageFile != null && !_isLoading) ...[
                 const SizedBox(height: 16),
                 _buildControlButtons(),
               ],
               
               const SizedBox(height: 48),
               _buildDisclaimer(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCropWarningBox() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
         color: const Color(0xFF00E5FF).withOpacity(0.08),
         borderRadius: BorderRadius.circular(20),
         border: Border.all(color: const Color(0xFF00E5FF).withOpacity(0.3)),
         boxShadow: [
           BoxShadow(
             color: const Color(0xFF00E5FF).withOpacity(0.02),
             blurRadius: 10,
             offset: const Offset(0, 4),
           )
         ]
      ),
      child: const Row(
         crossAxisAlignment: CrossAxisAlignment.start,
         children: [
            Icon(Icons.crop_free_rounded, color: Color(0xFF00E5FF), size: 28),
            SizedBox(width: 16),
            Expanded(
               child: Text(
                  "ÖNEMLİ: Kırpma ekranı açıldığında kareyi sadece BEN/LEKE'nin tam üzerine getirin. Çerçevede uzak arka plan bırakmayın!",
                  style: TextStyle(color: Color(0xFFE0F7FA), fontSize: 13, height: 1.4, fontWeight: FontWeight.w500),
               ),
            ),
         ],
      ),
    );
  }

  Widget _buildImagePreview() {
    return Container(
      height: 320,
      decoration: BoxDecoration(
        color: const Color(0xFF131B2F),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: const Color(0xFF00E5FF).withOpacity(0.2), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF00E5FF).withOpacity(0.05),
            blurRadius: 20,
            offset: const Offset(0, 10),
          )
        ]
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(28),
        child: _imageFile != null
            ? (kIsWeb 
                ? Image.network(_imageFile!.path, fit: BoxFit.cover) 
                : Image.file(File(_imageFile!.path), fit: BoxFit.cover))
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: const Color(0xFF00E5FF).withOpacity(0.1),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.document_scanner_outlined, size: 56, color: Color(0xFF00E5FF)),
                  ),
                  const SizedBox(height: 24),
                  const Text('Analiz İçin Görüntü Yükleyin', 
                    style: TextStyle(
                      color: Colors.white70, 
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 0.5,
                    )
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildControlButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
            icon: const Icon(Icons.camera_alt_rounded),
            label: const Text('Kamera', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: 0.5)),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 20),
              backgroundColor: const Color(0xFF00E5FF),
              foregroundColor: const Color(0xFF0A0E17),
              elevation: 8,
              shadowColor: const Color(0xFF00E5FF).withOpacity(0.6),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_rounded),
            label: const Text('Galeri', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: 0.5)),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 20),
              backgroundColor: const Color(0xFF131B2F),
              foregroundColor: Colors.white,
              elevation: 0,
              side: BorderSide(color: const Color(0xFF00E5FF).withOpacity(0.3), width: 1.5),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildResultsSection() {
    if (_isLoading) {
      return Column(
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(color: Theme.of(context).colorScheme.primary.withOpacity(0.2), blurRadius: 20)
              ]
            ),
            child: const CircularProgressIndicator(strokeWidth: 3),
          ),
          const SizedBox(height: 24),
          Text("Yapay Zeka Analiz Ediyor...", 
            style: TextStyle(
              fontSize: 16, 
              fontWeight: FontWeight.w600,
              color: Theme.of(context).colorScheme.primary
            )
          ),
        ],
      );
    }

    if (_imageFile == null) return const SizedBox.shrink();

    if (_underConfidence) {
      return Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: Colors.orange.withOpacity(0.08),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.orange.withOpacity(0.3), width: 1.5),
        ),
        child: const Column(
          children: [
            Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 56),
            SizedBox(height: 16),
            Text(
              "Lezyon Tam Olarak Tespit Edilemedi",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.orange),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 12),
            Text(
              "Lütfen alanı iyi aydınlatın, net odaklayın ve daha yakından tekrar çekim yapın.",
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey, height: 1.4),
            )
          ],
        ),
      );
    }

    if (_topResults.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text("Analiz Sonuçları", style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, letterSpacing: -0.5)),
            Row(
              children: [
                if (!_isSaved)
                  IconButton(
                    icon: const Icon(Icons.save_outlined),
                    color: Theme.of(context).colorScheme.primary,
                    style: IconButton.styleFrom(
                      backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                      padding: const EdgeInsets.all(12),
                    ),
                    onPressed: () => _saveToDatabase(_classNames[_topResults.first.key] ?? "Bilinmiyor"),
                    tooltip: "Geçmişe Kaydet",
                  )
                else
                  IconButton(
                    icon: const Icon(Icons.check_circle_rounded),
                    color: Colors.green,
                    style: IconButton.styleFrom(
                      backgroundColor: Colors.green.withOpacity(0.15),
                      padding: const EdgeInsets.all(12),
                    ),
                    onPressed: null,
                    tooltip: "Kaydedildi",
                  ),
                const SizedBox(width: 8),
                IconButton(
                  icon: const Icon(Icons.ios_share_rounded),
                  color: Theme.of(context).colorScheme.primary,
                  style: IconButton.styleFrom(
                    backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                    padding: const EdgeInsets.all(12),
                  ),
                  onPressed: _shareScreenshot,
                  tooltip: "Sonucu Doktora Gönder",
                ),
              ],
            )
          ],
        ),
        const SizedBox(height: 20),
        ..._topResults.map((result) {
          final label = result.key;
          final probability = result.value * 100;
          final riskLevel = _riskLevels[label]!;
          final riskColor = _riskColors[label]!;
          final fullName = _classNames[label]!;

          bool isFirst = result == _topResults.first;

          return Container(
            margin: const EdgeInsets.only(bottom: 14),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
            decoration: BoxDecoration(
              gradient: isFirst ? LinearGradient(
                colors: [riskColor.withOpacity(0.15), riskColor.withOpacity(0.05)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ) : null,
              color: isFirst ? null : Theme.of(context).colorScheme.surface,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: isFirst ? riskColor.withOpacity(0.5) : Theme.of(context).colorScheme.outlineVariant.withOpacity(0.5),
                width: isFirst ? 2 : 1,
              ),
              boxShadow: isFirst ? [
                BoxShadow(
                  color: riskColor.withOpacity(0.1),
                  blurRadius: 15,
                  offset: const Offset(0, 5),
                )
              ] : [
                BoxShadow(
                  color: Colors.black.withOpacity(0.02),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                )
              ],
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(fullName, style: TextStyle(fontSize: isFirst ? 18 : 16, fontWeight: isFirst ? FontWeight.bold : FontWeight.w600)),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: riskColor.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Container(width: 8, height: 8, decoration: BoxDecoration(color: riskColor, shape: BoxShape.circle)),
                                const SizedBox(width: 6),
                                Text(riskLevel, style: TextStyle(color: riskColor, fontWeight: FontWeight.bold, fontSize: 13)),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                Text(
                  "%${probability.toStringAsFixed(1)}",
                  style: TextStyle(
                    fontSize: isFirst ? 26 : 20, 
                    fontWeight: FontWeight.w900, 
                    color: isFirst ? riskColor : Theme.of(context).colorScheme.onSurface.withOpacity(0.7)
                  ),
                ),
              ],
            ),
          );
        }).toList(),
        const SizedBox(height: 24),
        if (_camImageBase64 != null) ...[
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.secondaryContainer.withOpacity(0.3),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Theme.of(context).colorScheme.secondary.withOpacity(0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.hub_rounded, color: Theme.of(context).colorScheme.secondary),
                    const SizedBox(width: 8),
                    const Text("Modelin Odaklandığı Alanlar", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  ],
                ),
                const SizedBox(height: 12),
                const Text("Kırmızı ve sıcak renkli bölgeler, yapay zekanın karar verirken lezyon üzerinde en çok dikkat ettiği alanları gösterir.", style: TextStyle(fontSize: 13, color: Colors.grey)),
                const SizedBox(height: 16),
                ClipRRect(
                  borderRadius: BorderRadius.circular(16),
                  child: Image.memory(base64Decode(_camImageBase64!), fit: BoxFit.cover, width: double.infinity),
                ),
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Theme.of(context).colorScheme.outlineVariant.withOpacity(0.5)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Neden ${_classNames[_topResults.first.key]}?", style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                      const SizedBox(height: 6),
                      Text(_classExplanations[_topResults.first.key] ?? '', style: TextStyle(fontSize: 13, height: 1.4, color: Theme.of(context).colorScheme.onSurfaceVariant)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ] else ...[
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _isExplaining ? null : _explainResult,
              icon: _isExplaining 
                  ? SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Theme.of(context).colorScheme.onPrimaryContainer)) 
                  : const Icon(Icons.hub_rounded),
              label: Text(_isExplaining ? 'Açıklama Üretiliyor...' : 'Yapay Zeka Nereye Odaklandı? (Açıkla)', style: const TextStyle(fontWeight: FontWeight.bold)),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Theme.of(context).colorScheme.primaryContainer,
                foregroundColor: Theme.of(context).colorScheme.onPrimaryContainer,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              ),
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.errorContainer.withOpacity(0.4),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Theme.of(context).colorScheme.error.withOpacity(0.2)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.medical_information_rounded, color: Theme.of(context).colorScheme.error, size: 28),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              "DİKKAT: Bu uygulama yalnızca bilgilendirme amaçlıdır ve tıbbi tanı yerine geçmez. Şüpheli durumlarda vakit kaybetmeden bir Dermatoloğa başvurunuz.",
              style: TextStyle(
                color: Theme.of(context).colorScheme.onErrorContainer, 
                fontSize: 13, 
                height: 1.5,
                fontWeight: FontWeight.w500
              ),
            ),
          ),
        ],
      ),
    );
  }
}
