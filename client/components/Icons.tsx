import { Info, LucideIcon, MoonStar, Sun, Check, ChevronDown, ChevronUp } from 'lucide-react-native';
import { cssInterop } from 'nativewind';

function interopIcon(icon: LucideIcon) {
  cssInterop(icon, {
    className: {
      target: 'style',
      nativeStyleToProp: {
        color: true,
        opacity: true,
      },
    },
  });
}

interopIcon(Info);
interopIcon(MoonStar);
interopIcon(Sun);
interopIcon(Check);
interopIcon(ChevronDown);
interopIcon(ChevronUp);

export { 
  Info, 
  MoonStar, 
  Sun,
  Check,
  ChevronDown,
  ChevronUp,
};
